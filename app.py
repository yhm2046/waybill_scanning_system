import streamlit as st
import pandas as pd
import json
import io
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from datetime import datetime
from PIL import Image
from pydantic import BaseModel, Field
from typing import List

# ==========================================
# 1. 定義固定資料結構
# ==========================================
class DeliveryItem(BaseModel):
    item_name: str = Field(description="貨品名稱")
    quantity: int = Field(description="數量")
    unit_price: float = Field(description="單價")
    subtotal: float = Field(description="小計金額")
    is_unclear: bool = Field(description="字跡是否模糊或手寫難以辨認")

class DeliveryNote(BaseModel):
    doc_number: str = Field(description="送貨單號")
    date: str = Field(description="送貨日期 (YYYY-MM-DD)")
    customer_name: str = Field(description="收貨客戶名稱")
    items: List[DeliveryItem] = Field(description="貨品明細清單")
    stated_total: float = Field(description="單據上印製或手寫的總金額")

# ==========================================
# 2. 固定規則檢查引擎
# ==========================================
def validate_delivery_note(data: dict, edited_df: pd.DataFrame, stated_total: float) -> List[str]:
    warnings = []

    if not str(data.get("doc_number", "")).strip():
        warnings.append("🚨 【欄位缺失】找不到「送貨單號」，請人工補填。")
    if not str(data.get("customer_name", "")).strip():
        warnings.append("🚨 【欄位缺失】找不到「收貨客戶名稱」，請人工補填。")

    calculated_total = 0.0
    for idx, row in edited_df.iterrows():
        qty = float(row.get("quantity", 0))
        price = float(row.get("unit_price", 0.0))
        sub = float(row.get("subtotal", 0.0))
        expected_sub = round(qty * price, 2)
        calculated_total += sub

        if abs(expected_sub - sub) > 0.01:
            warnings.append(
                f"⚠️ 【明細算術不符】第 {idx + 1} 行「{row.get('item_name')}」："
                f"數量({qty}) × 單價({price}) 應為 {expected_sub}，但單據小計為 {sub}。"
            )
        if row.get("is_unclear", False):
            warnings.append(
                f"👁️ 【手寫/字跡模糊】第 {idx + 1} 行「{row.get('item_name')}」包含手寫塗改或模糊字跡，請對照左側原圖確認。"
            )

    if abs(round(calculated_total, 2) - round(stated_total, 2)) > 0.01:
        warnings.append(
            f"💰 【總金額異常】明細加總金額為 ${calculated_total:.2f}，"
            f"與單據標示總金額 ${stated_total:.2f} 不一致！請人工核對。"
        )
    return warnings

# ==========================================
# 3. AI 識別層（提速 + 自動重試 + 自動換模型）
# ==========================================
CACHE_DIR = Path("ai_cache")
CACHE_DIR.mkdir(exist_ok=True)

PROMPT = (
    "讀取這張物流送貨單或發票圖片，按指定格式提取：單號、日期(YYYY-MM-DD)、"
    "客戶名稱、貨品明細、總金額。手寫塗改或模糊的明細行，is_unclear 設為 true。"
    "只輸出結果，不要解釋。"
)

# 這些關鍵字出現在錯誤訊息裡，代表「稍後重試或換模型」可能有用
RETRYABLE = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand",
             "overloaded", "DEADLINE", "timed out", "timeout", "Timeout")

def prepare_image(image: Image.Image, max_side: int) -> bytes:
    """縮小並轉成 JPEG：體積小，上傳快"""
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()

def make_thinking_config(types, model: str):
    """把『思考』調到最低。Gemini 3 與 2.5 的設定方式不同，兩種不能同時用"""
    m = model.lower()
    try:
        if "gemini-3" in m:
            return types.ThinkingConfig(thinking_level="minimal" if "flash" in m else "low")
        if "gemini-2.5" in m:
            return types.ThinkingConfig(thinking_budget=128 if "pro" in m else 0)
    except Exception:
        pass
    return None

def call_gemini(img_bytes: bytes, api_key: str, model: str, timeout_s: int) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

    def run(thinking):
        cfg = dict(
            response_mime_type="application/json",
            response_schema=DeliveryNote,
            max_output_tokens=4096,
        )                       # 不設 temperature（Gemini 3 建議保持預設）
        if thinking is not None:
            cfg["thinking_config"] = thinking
        return client.models.generate_content(
            model=model,
            contents=[image_part, PROMPT],
            config=types.GenerateContentConfig(**cfg),
        )

    thinking = make_thinking_config(types, model)
    try:
        resp = run(thinking)
    except Exception as e:
        if thinking is not None and "think" in str(e).lower():
            resp = run(None)    # 這個模型不接受該設定，退回預設
        else:
            raise
    return json.loads(resp.text)

def call_with_fallback(img_bytes, api_key, models, timeout_s, budget_s, hedge_s=8):
    """備援競速：先送第一個模型；若 hedge_s 秒內沒回應（或很快失敗），
    不等它超時，立刻同時再送下一個模型，誰先成功就用誰。總時間受 budget_s 限制。"""
    log, deadline = [], time.time() + budget_s
    seq = list(models) + [models[0]]        # 最後再補一路同模型（換一條後端線路，常常就不卡了）
    ex = ThreadPoolExecutor(max_workers=len(seq))
    pending, state = {}, {"i": 0, "last": 0.0}

    def launch():
        m = seq[state["i"]]
        state["i"] += 1
        rem = deadline - time.time()
        f = ex.submit(call_gemini, img_bytes, api_key, m, max(5, min(timeout_s, rem)))
        pending[f] = (m, time.time())
        state["last"] = time.time()
        log.append(f"送出 {m}（第 {state['i']} 路，已過 {budget_s - rem:.0f} 秒）")

    try:
        launch()
        while True:
            rem = deadline - time.time()
            if rem <= 0:
                raise RuntimeError("已超過總等待上限，停止嘗試。\n" + "\n".join(log))
            more = state["i"] < len(seq)
            wait_t = min(rem, max(0.1, hedge_s - (time.time() - state["last"]))) if more else rem
            done, _ = wait(list(pending), timeout=wait_t, return_when=FIRST_COMPLETED)
            for f in done:
                m, t = pending.pop(f)
                try:
                    data = f.result()
                    log.append(f"{m}：成功（{time.time() - t:.1f} 秒）")
                    return data, m, log
                except Exception as e:
                    msg = str(e)
                    log.append(f"{m}：失敗（{time.time() - t:.1f} 秒）{msg[:90]}")
                    if "API_KEY_INVALID" in msg or "API key not valid" in msg:
                        raise RuntimeError("API Key 無效，請檢查。")
            more = state["i"] < len(seq)
            if not pending and not more:
                raise RuntimeError("所有模型都失敗了：\n" + "\n".join(log))
            if more and deadline - time.time() > 3 and (done or time.time() - state["last"] >= hedge_s):
                launch()
    finally:
        ex.shutdown(wait=False)             # 卡住的請求不再等，讓它自己超時結束

def extract_from_image(image, api_key, models, max_side, timeout_s, budget_s, use_cache, hedge_s=8):
    t0 = time.time()
    img_bytes = prepare_image(image, max_side)
    stats = {"prepare": time.time() - t0, "kb": len(img_bytes) // 1024,
             "api": 0.0, "cached": False, "model": "", "log": []}

    cache_file = CACHE_DIR / (hashlib.md5(img_bytes).hexdigest() + ".json")
    if use_cache and cache_file.exists():
        stats["cached"] = True
        obj = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "_meta" in obj and "data" in obj:   # 新格式：含當時的模型與耗時
            stats["model"] = obj["_meta"].get("model", "")
            stats["orig_api"] = obj["_meta"].get("api")
            return obj["data"], stats
        return obj, stats                                                 # 舊格式快取：沒有模型資訊

    if not api_key:
        raise ValueError("請先在左側輸入 Gemini API Key。")
    if not models:
        raise ValueError("請在左側填寫至少一個模型名稱。")

    t1 = time.time()
    data, used, log = call_with_fallback(img_bytes, api_key, models, timeout_s, budget_s, hedge_s)
    stats.update({"api": time.time() - t1, "model": used, "log": log})
    cache_file.write_text(json.dumps({"_meta": {"model": used, "api": round(stats["api"], 1)}, "data": data},
                                     ensure_ascii=False), encoding="utf-8")
    return data, stats

def test_key(api_key, models):
    """用列出模型清單的方式測試 Key（不消耗識別額度）。回傳 (是否可用, 說明文字)"""
    if not api_key:
        return False, "還沒有輸入 API Key。"
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        return False, f"沒有安裝 google-genai，或 Python 版本太舊：{e}"
    try:
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=15000))
        names = [m.name.replace("models/", "") for m in client.models.list()]
    except Exception as e:
        msg = str(e)
        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
            return False, "API Key 無效：請重新複製，或到 Google AI Studio 建立新的 Key。"
        if any(k in msg for k in ("timed out", "timeout", "Timeout", "Connect", "proxy", "Proxy")):
            return False, f"連不上 Google（網絡或代理問題）：{msg[:120]}"
        return False, f"測試失敗：{msg[:160]}"
    missing = [m for m in models if m not in names]
    text = f"Key 可用，共 {len(names)} 個模型。"
    if missing:
        flash = [n for n in names if "flash" in n and "image" not in n][:6]
        text += f"\n\n⚠️ 你填的這些模型名稱不存在：{', '.join(missing)}"
        if flash:
            text += f"\n可用的 flash 模型例如：{', '.join(flash)}"
    else:
        text += " 你填的模型名稱都存在。"
    return True, text

# ==========================================
# 4. 批次處理（最多 5 張同時送出，總時間≈最慢那一張）
# ==========================================
MAX_FILES = 5

def run_batch(blobs, api_key, models, max_side, timeout_s, budget_s, use_cache, progress, hedge_s=8):
    """blobs: {key: bytes}。同時送出所有圖片，每張各自重試/換模型；錯誤只影響那一張。"""
    results, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=max(1, len(blobs))) as ex:
        futs = {}
        for k, b in blobs.items():
            img = Image.open(io.BytesIO(b))
            futs[ex.submit(extract_from_image, img, api_key, models,
                           max_side, timeout_s, budget_s, use_cache, hedge_s)] = k
        done = 0
        for f in as_completed(futs):
            k = futs[f]
            try:
                data, stats = f.result()
                results[k] = {"ok": True, "raw": data, "stats": stats}
            except Exception as e:
                results[k] = {"ok": False, "error": str(e)}
            done += 1
            progress.progress(done / len(futs), text=f"已完成 {done}/{len(futs)} 張")
    return results, time.time() - t0

def quick_warnings(data):
    df = pd.DataFrame(data.get("items", []))
    return validate_delivery_note(data, df, float(data.get("stated_total", 0.0)))

# ==========================================
# 5. Streamlit 網頁互動介面
# ==========================================
st.set_page_config(page_title="LOGFLOWS 單據識別與驗證 Demo", layout="wide")
st.title("📦 物流單據 AI 批次識別與防呆驗證 Demo")
st.caption(f"一次上傳最多 {MAX_FILES} 張 ➔ 同時送 AI 識別 ➔ 固定規則自動檢查 ➔ 逐張人工確認 ➔ 匯出")

with st.sidebar:
    st.header("⚙️ 運行設定")
    api_key = st.text_input("輸入 Gemini API Key：", type="password", help="可於 Google AI Studio 取得")
    api_key = api_key.strip().strip('"').strip("'")
    models_text = st.text_input(
        "模型名稱（可填多個，逗號隔開，按順序嘗試）",
        value="gemini-3.5-flash, gemini-3.8-flash, gemini-2.5-flash",
        help="名稱請以你的 Key 實際可用的為準",
    )
    models = [m.strip() for m in models_text.split(",") if m.strip()]
    max_side = st.slider("圖片最長邊（像素）", 800, 1600, 1120, 160, help="越小越快，但字太小可能讀錯")
    hedge_s = st.number_input("備援等待（秒）", 3, 30, 8, 1,
                              help="第一個模型超過這麼多秒沒回應，就同時再送一路備援，誰先好用誰")
    timeout_s = st.number_input("單次請求超時（秒）", 10, 300, 25, 5)
    budget_s = st.number_input("每張總等待上限（秒）", 20, 600, 30, 5)
    use_cache = st.checkbox("同一張圖片使用上次識別結果（快取）", value=True)

    if st.button("🔑 測試 Key", use_container_width=True):
        with st.spinner("測試中…"):
            ok, msg = test_key(api_key, models)
        (st.success if ok else st.error)(msg)

    st.divider()
    st.markdown("**💡 設計原則：**\n1. 5 張同時送出，不是一張一張排隊\n2. 思考調到最低、圖片壓縮\n3. 某一張失敗不影響其他張\n4. 算術與格式由 Python 固定規則把關\n5. 涉及金額與模糊字跡強制人工確認")

uploaded = st.file_uploader(f"① 請上傳送貨單或發票圖片（最多 {MAX_FILES} 張，JPG / PNG）",
                            type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if len(uploaded) > MAX_FILES:
    st.warning(f"一次最多 {MAX_FILES} 張，只處理前 {MAX_FILES} 張。")
    uploaded = uploaded[:MAX_FILES]

# 每張圖用「序號 + 內容雜湊」當 key，避免同名檔案互相覆蓋
blobs = {}
for i, f in enumerate(uploaded):
    b = f.getvalue()
    blobs[f"{i}_{hashlib.md5(b).hexdigest()[:8]}"] = b
names = {k: uploaded[i].name for i, k in enumerate(blobs)}

if blobs:
    if st.button(f"🔍 ② 同時識別這 {len(blobs)} 張", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="已送出，等待結果…")
        results, wall = run_batch(blobs, api_key, models, max_side, timeout_s, budget_s, use_cache, bar, hedge_s)
        bar.empty()
        st.session_state["batch"] = {"results": results, "wall": wall, "keys": list(blobs)}

batch = st.session_state.get("batch")
if blobs and batch and set(batch["keys"]) == set(blobs):
    results = batch["results"]
    ok_n = sum(1 for r in results.values() if r["ok"])
    wall = batch["wall"]
    n_cache = sum(1 for r in results.values() if r["ok"] and r["stats"]["cached"])
    n_fresh = ok_n - n_cache
    n_fail = len(results) - ok_n
    line = f"共 {len(results)} 張：🆕 新識別 {n_fresh} 張｜⚡ 快取 {n_cache} 張｜❌ 失敗 {n_fail} 張"
    if n_fail == len(results):
        st.error(line + "。全部失敗，請先按左側「🔑 測試 Key」。")
    elif n_fresh == 0:
        st.warning(line + "。這次沒有真正呼叫 AI，結果都來自舊記錄。")
    else:
        (st.success if wall <= 30 else st.warning)(
            f"{line}｜⏱️ 本次總耗時 {wall:.1f} 秒"
            + ("" if wall <= 30 else "（超過 30 秒，可調小圖片尺寸或檢查模型是否太忙）"))

    summary_ph = st.empty()      # 總覽表等逐張核對區處理完再填，這樣異常數會反映你修改後的內容
    live_rows = []

    # ---- 逐張核對 ----
    st.subheader("③ 逐張人工核對")
    exports = []
    tabs = st.tabs([f"{i + 1}. {names[k]}" for i, k in enumerate(blobs)])
    for tab, k in zip(tabs, blobs):
        with tab:
            r = results.get(k)
            col_img, col_data = st.columns([1, 1.3])
            with col_img:
                st.image(Image.open(io.BytesIO(blobs[k])), use_container_width=True)
            with col_data:
                if r is None or not r["ok"]:
                    err = r["error"] if r else "沒有結果"
                    st.error(f"識別失敗：{err}")
                    live_rows.append({"檔案": names[k], "狀態": "❌ 失敗", "單號": "-", "待處理異常": "-",
                                      "AI 耗時": "-", "模型": "-", "說明": err.splitlines()[0][:40]})
                    continue
                data = r["raw"]
                s = r["stats"]
                if s["cached"]:
                    st.info("⚡ 結果來自同一張圖片先前的識別（快取）。")
                else:
                    st.caption(f"⏱️ 圖片處理 {s['prepare']:.1f} 秒｜AI {s['api']:.1f} 秒｜"
                               f"上傳 {s['kb']} KB｜{s['model']}")
                    with st.expander("查看嘗試記錄"):
                        for line in s["log"]:
                            st.write(line)

                c1, c2, c3 = st.columns(3)
                doc_num = c1.text_input("送貨單號", value=data.get("doc_number", ""), key=f"num_{k}")
                doc_date = c2.text_input("單據日期", value=data.get("date", ""), key=f"date_{k}")
                cust = c3.text_input("客戶名稱", value=data.get("customer_name", ""), key=f"cust_{k}")

                edited_df = st.data_editor(
                    pd.DataFrame(data.get("items", [])),
                    column_config={
                        "item_name": "貨品/服務名稱",
                        "quantity": st.column_config.NumberColumn("數量", min_value=0, step=1),
                        "unit_price": st.column_config.NumberColumn("單價 ($)", format="$%.2f"),
                        "subtotal": st.column_config.NumberColumn("小計 ($)", format="$%.2f"),
                        "is_unclear": st.column_config.CheckboxColumn("手寫/模糊待確認?"),
                    },
                    num_rows="dynamic", use_container_width=True, key=f"df_{k}")
                total = st.number_input("單據標示總金額 ($)", value=float(data.get("stated_total", 0.0)),
                                        step=10.0, key=f"tot_{k}")

                warns = validate_delivery_note(
                    {"doc_number": doc_num, "date": doc_date, "customer_name": cust}, edited_df, total)
                if warns:
                    st.error(f"發現 {len(warns)} 項異常或待確認項目：")
                    for w in warns:
                        st.write(w)
                else:
                    st.success("✅ 格式與金額交叉驗算全部通過！")

                if s["cached"]:
                    t_txt = f"快取（原 {s['orig_api']} 秒）" if s.get("orig_api") is not None else "快取"
                    m_txt = s["model"] or "不明（舊快取）"
                else:
                    t_txt, m_txt = f"{s['api']:.1f} 秒", s["model"]
                live_rows.append({"檔案": names[k], "狀態": "⚡ 快取" if s["cached"] else "🆕 新識別",
                                  "單號": doc_num or "（空）", "待處理異常": str(len(warns)),
                                  "AI 耗時": t_txt, "模型": m_txt, "說明": ""})

                if st.checkbox("✍️ 我已核對這張單據的金額與欄位無誤", key=f"ok_{k}"):
                    exports.append({
                        "source_file": names[k], "doc_number": doc_num, "date": doc_date,
                        "customer_name": cust, "stated_total": total,
                        "verified_items": edited_df.to_dict(orient="records"),
                        "audit_log": {
                            "verified_by": "Operator_Manual_Check",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "remaining_warnings_count": len(warns),
                            "ai_original_snapshot": data}})

    summary_ph.dataframe(pd.DataFrame(live_rows).astype(str), use_container_width=True, hide_index=True)

    # ---- 匯出（只含已勾選確認的） ----
    st.divider()
    st.subheader("④ 匯出已確認的單據")
    st.write(f"已確認 {len(exports)} / {ok_n} 張")
    if exports:
        st.download_button("💾 下載已確認單據 JSON（模擬寫入 TMS）",
                           data=json.dumps(exports, ensure_ascii=False, indent=2),
                           file_name=f"verified_batch_{datetime.now():%Y%m%d_%H%M%S}.json",
                           mime="application/json")