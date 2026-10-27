import streamlit as st
import pandas as pd
import json
import io
import time
import hashlib
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

def call_with_fallback(img_bytes, api_key, models, timeout_s, budget_s, retries=2):
    """依序嘗試多個模型；太忙/超時就等幾秒重試，仍不行換下一個；總時間有上限"""
    log, deadline = [], time.time() + budget_s
    for model in models:
        for attempt in range(1, retries + 1):
            remaining = deadline - time.time()
            if remaining <= 3:
                raise RuntimeError("已超過總等待上限，停止嘗試。\n" + "\n".join(log))
            t = time.time()
            try:
                data = call_gemini(img_bytes, api_key, model, min(timeout_s, remaining))
                log.append(f"{model} 第 {attempt} 次：成功（{time.time() - t:.1f} 秒）")
                return data, model, log
            except Exception as e:
                msg = str(e)
                log.append(f"{model} 第 {attempt} 次：失敗（{time.time() - t:.1f} 秒）{msg[:90]}")
                if "API_KEY_INVALID" in msg or "API key not valid" in msg:
                    raise RuntimeError("API Key 無效，請檢查。")
                if not any(k in msg for k in RETRYABLE):
                    break                       # 這個模型有別的問題（例如名稱不存在），換下一個
                time.sleep(2 * attempt)
    raise RuntimeError("所有模型都失敗了：\n" + "\n".join(log))

def extract_from_image(image, api_key, models, max_side, timeout_s, budget_s, use_cache):
    t0 = time.time()
    img_bytes = prepare_image(image, max_side)
    stats = {"prepare": time.time() - t0, "kb": len(img_bytes) // 1024,
             "api": 0.0, "cached": False, "model": "", "log": []}

    cache_file = CACHE_DIR / (hashlib.md5(img_bytes).hexdigest() + ".json")
    if use_cache and cache_file.exists():
        stats["cached"] = True
        return json.loads(cache_file.read_text(encoding="utf-8")), stats

    if not api_key:
        raise ValueError("請先在左側輸入 Gemini API Key。")
    if not models:
        raise ValueError("請在左側填寫至少一個模型名稱。")

    t1 = time.time()
    data, used, log = call_with_fallback(img_bytes, api_key, models, timeout_s, budget_s)
    stats.update({"api": time.time() - t1, "model": used, "log": log})
    cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data, stats

# ==========================================
# 4. Streamlit 網頁互動介面
# ==========================================
st.set_page_config(page_title="LOGFLOWS 單據識別與驗證 Demo", layout="wide")
st.title("📦 物流單據 AI 識別與防呆驗證 Demo")
st.caption("上傳單據 ➔ AI 快速提取 ➔ 固定規則自動檢查 ➔ 人工確認與匯出")

with st.sidebar:
    st.header("⚙️ 運行設定")
    api_key = st.text_input("輸入 Gemini API Key：", type="password", help="可於 Google AI Studio 取得")
    api_key = api_key.strip().strip('"').strip("'")
    models_text = st.text_input(
        "模型名稱（可填多個，逗號隔開，按順序嘗試）",
        value="gemini-3.5-flash, gemini-3.8-flash, gemini-2.5-flash",
        help="名稱請以你的 Key 實際可用的為準（用 curl 列出模型清單查看）",
    )
    models = [m.strip() for m in models_text.split(",") if m.strip()]
    max_side = st.slider("圖片最長邊（像素）", 800, 1600, 1280, 160, help="越小越快，但字太小可能讀錯")
    timeout_s = st.number_input("單次請求超時（秒）", 10, 300, 30, 5)
    budget_s = st.number_input("總等待上限（秒）", 20, 600, 90, 10)
    use_cache = st.checkbox("同一張圖片使用上次識別結果（快取）", value=True)

    st.divider()
    st.markdown("**💡 設計原則：**\n1. 思考調到最低、圖片壓縮，縮短等待\n2. 模型太忙時自動重試、換模型\n3. 算術與格式由 Python 固定規則把關\n4. 涉及金額與模糊字跡強制人工確認")

uploaded_file = st.file_uploader("① 請上傳送貨單或發票圖片 (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col_img, col_data = st.columns([1, 1.3])

    with col_img:
        st.subheader("📄 原始單據影像")
        st.image(img, use_container_width=True)

        if st.button("🔍 ② 執行 AI 快速讀取與檢查", type="primary", use_container_width=True):
            try:
                with st.spinner("AI 正在讀取單據…（模型太忙時會自動重試或換模型）"):
                    raw_result, stats = extract_from_image(
                        img, api_key, models, max_side, timeout_s, budget_s, use_cache)
                st.session_state["ai_raw_output"] = raw_result
                st.session_state["parsed_data"] = json.loads(json.dumps(raw_result))
                st.session_state["stats"] = stats
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"識別失敗：{e}")

    with col_data:
        if "parsed_data" in st.session_state:
            data = st.session_state["parsed_data"]
            stats = st.session_state.get("stats", {})
            if stats.get("cached"):
                st.info("⚡ 此結果來自同一張圖片先前的實際識別（快取），這次沒有重新呼叫 AI。")
            elif stats:
                st.caption(f"⏱️ 圖片處理 {stats['prepare']:.1f} 秒｜AI 識別 {stats['api']:.1f} 秒｜"
                           f"上傳 {stats['kb']} KB｜使用模型：{stats['model']}")
                with st.expander("查看嘗試記錄"):
                    for line in stats.get("log", []):
                        st.write(line)

            st.subheader("📋 ③ AI 提取結果與人工核對區")
            c1, c2, c3 = st.columns(3)
            doc_num = c1.text_input("送貨單號", value=data.get("doc_number", ""))
            doc_date = c2.text_input("單據日期", value=data.get("date", ""))
            cust_name = c3.text_input("客戶名稱", value=data.get("customer_name", ""))

            st.markdown("**貨品明細表（可直接修改數值，或取消勾選模糊標記）：**")
            df_items = pd.DataFrame(data.get("items", []))
            edited_df = st.data_editor(
                df_items,
                column_config={
                    "item_name": "貨品/服務名稱",
                    "quantity": st.column_config.NumberColumn("數量", min_value=0, step=1),
                    "unit_price": st.column_config.NumberColumn("單價 ($)", format="$%.2f"),
                    "subtotal": st.column_config.NumberColumn("小計 ($)", format="$%.2f"),
                    "is_unclear": st.column_config.CheckboxColumn("手寫/模糊待確認?"),
                },
                num_rows="dynamic",
                use_container_width=True,
            )

            stated_total = st.number_input("單據標示總金額 ($)", value=float(data.get("stated_total", 0.0)), step=10.0)

            current_main = {"doc_number": doc_num, "date": doc_date, "customer_name": cust_name}
            warnings = validate_delivery_note(current_main, edited_df, stated_total)

            st.divider()
            st.subheader("🛡️ ④ 系統自動檢查報告 (Rule Engine)")
            if warnings:
                st.error(f"發現 {len(warnings)} 項異常或待確認項目，請人工修正上方表格：")
                for w in warnings:
                    st.write(w)
            else:
                st.success("✅ 格式與金額交叉驗算全部通過！")

            st.divider()
            human_confirmed = st.checkbox("✍️ 我已作為操作人員核對上述單據金額與欄位無誤，授權匯入系統")

            if st.button("📤 ⑤ 確認並匯出標準 JSON (模擬寫入 TMS 介面)", disabled=not human_confirmed):
                final_payload = {
                    "doc_number": doc_num,
                    "date": doc_date,
                    "customer_name": cust_name,
                    "stated_total": stated_total,
                    "verified_items": edited_df.to_dict(orient="records"),
                    "audit_log": {
                        "verified_by": "Operator_Manual_Check",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "remaining_warnings_count": len(warnings),
                        "ai_original_snapshot": st.session_state["ai_raw_output"],
                    },
                }
                st.success("🎉 資料已成功打包！")
                st.json(final_payload)

                json_str = json.dumps(final_payload, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 下載 JSON 檔案",
                    data=json_str,
                    file_name=f"verified_{doc_num}.json",
                    mime="application/json",
                )