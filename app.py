import io
import json
import hashlib
import statistics
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from core import validate_delivery_note, run_batch, quick_warnings, test_key

MAX_FILES = 50          # 一次最多上傳幾張
HEDGE_MAX_IMAGES = 5    # 超過這個張數，自動關閉備援（避免請求翻倍、更容易被限流）

st.set_page_config(page_title="LOGFLOWS 單據識別與驗證 Demo", layout="wide")
st.title("📦 物流單據 AI 批次識別與防呆驗證 Demo")
st.caption("上傳 ➔ 固定並發數同時識別 ➔ 規則自動檢查 ➔ 只有異常的單據需要人工處理 ➔ 匯出")

# ---------------- 側邊欄 ----------------
with st.sidebar:
    st.header("⚙️ 運行設定")
    api_key = st.text_input("輸入 Gemini API Key：", type="password", help="可於 Google AI Studio 取得")
    api_key = api_key.strip().strip('"').strip("'")
    models_text = st.text_input(
        "模型名稱（可填多個，逗號隔開，按順序嘗試）",
        value="gemini-3.1-flash-lite, gemini-3.5-flash-lite",
        help="名稱請以你的 Key 實際可用的為準")
    models = [m.strip() for m in models_text.split(",") if m.strip()]

    concurrency = st.number_input("並發數（同時處理幾張）", 1, 50, 10, 1,
                                  help="太大會被 Google 限流（429）。建議先 10，穩定再加到 20、50")
    max_side = st.slider("圖片最長邊（像素）", 800, 1600, 1120, 160, help="越小越快，但字太小可能讀錯")
    hedge_on = st.checkbox(f"小批量時開啟備援競速（超過 {HEDGE_MAX_IMAGES} 張自動關閉）", value=True)
    hedge_s = st.number_input("備援等待（秒）", 3, 30, 8, 1)
    timeout_s = st.number_input("單次請求超時（秒）", 10, 300, 15, 1,
                                help="正常約 5 秒；設短一點，卡住的請求可以更快換模型重試")
    rpm = st.number_input("每分鐘最多送出幾張（0=不限）", 0, 600, 0, 5,
                          help="被限流（429）時填你的 Key 額度，程式會自動排開請求")
    budget_s = st.number_input("每張總等待上限（秒）", 20, 600, 30, 5)
    use_cache = st.checkbox("同一張圖片使用上次識別結果（快取）", value=True,
                            help="做速度測試時請取消勾選，否則快取的圖片不會真的呼叫 AI")

    if st.button("🔑 測試 Key", use_container_width=True):
        with st.spinner("測試中…"):
            ok, msg = test_key(api_key, models)
        (st.success if ok else st.error)(msg)

    st.divider()
    st.markdown("**💡 設計原則：**\n1. 固定並發數的隊列，不是一次全部發出\n2. 大批量自動關閉備援，減少限流\n"
                "3. 某一張失敗不影響其他張\n4. 算術與格式由 Python 固定規則把關\n"
                "5. 只有異常、失敗的單據才需要人工逐張核對")

# ---------------- 上傳 ----------------
uploaded = st.file_uploader(f"① 請上傳送貨單或發票圖片（最多 {MAX_FILES} 張，JPG / PNG）",
                            type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if len(uploaded) > MAX_FILES:
    st.warning(f"一次最多 {MAX_FILES} 張，只取前 {MAX_FILES} 張。")
    uploaded = uploaded[:MAX_FILES]

n_run = 0
if uploaded:
    n_run = int(st.number_input(
        "本次處理前幾張（建議先測 10，再測 20，最後 50）",
        1, len(uploaded), min(10, len(uploaded)), 1))
    uploaded = uploaded[:n_run]

blobs = {}
for i, f in enumerate(uploaded):
    b = f.getvalue()
    blobs[f"{i}_{hashlib.md5(b).hexdigest()[:8]}"] = b
names = {k: uploaded[i].name for i, k in enumerate(blobs)}

eff_hedge = hedge_s if (hedge_on and len(blobs) <= HEDGE_MAX_IMAGES) else 0
eff_conc = min(int(concurrency), max(1, len(blobs)))
if blobs:
    st.caption(f"本次設定：處理 {len(blobs)} 張｜並發 {eff_conc}｜備援 "
               f"{'關閉' if eff_hedge == 0 else f'{eff_hedge} 秒'}｜快取 {'開' if use_cache else '關'}")
    if st.button(f"🔍 ② 開始識別這 {len(blobs)} 張", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="已送出，等待結果…")
        results, wall = run_batch(blobs, api_key, models, max_side, timeout_s, budget_s, use_cache,
                                  bar, eff_hedge, eff_conc, int(rpm))
        bar.empty()
        st.session_state["batch"] = {"results": results, "wall": wall, "keys": list(blobs),
                                     "conc": eff_conc, "hedge": eff_hedge}

batch = st.session_state.get("batch")
if blobs and batch and set(batch["keys"]) == set(blobs):
    results, wall = batch["results"], batch["wall"]

    # ---------------- 統計 ----------------
    ok_keys = [k for k in blobs if results[k]["ok"]]
    fresh = [k for k in ok_keys if not results[k]["stats"]["cached"]]
    cached = [k for k in ok_keys if results[k]["stats"]["cached"]]
    failed = [k for k in blobs if not results[k]["ok"]]
    n429 = sum(results[k]["stats"].get("n429", 0) for k in ok_keys) + \
        sum(results[k]["error"].count("429") for k in failed)
    lat = sorted(results[k]["stats"]["api"] for k in fresh)

    line = (f"共 {len(blobs)} 張：🆕 新識別 {len(fresh)}｜⚡ 快取 {len(cached)}｜❌ 失敗 {len(failed)}"
            f"｜並發 {batch['conc']}｜備援 {'關' if batch['hedge'] == 0 else '開'}")
    if len(failed) == len(blobs):
        st.error(line + "。全部失敗，請先按左側「🔑 測試 Key」。")
    elif not fresh:
        st.warning(line + "。這次沒有真正呼叫 AI，結果都來自舊記錄（速度測試請取消快取）。")
    else:
        detail = f"⏱️ 總耗時 {wall:.1f} 秒"
        if lat:
            detail += f"｜單張中位數 {statistics.median(lat):.1f} 秒、最慢 {lat[-1]:.1f} 秒"
        detail += f"｜遇到限流(429) {n429} 次"
        good = wall <= 30 and not failed
        (st.success if good else st.warning)(
            f"{line}\n\n{detail}｜" + ("✅ 30 秒內全部完成" if good else "❌ 沒有全部在 30 秒內完成"))
        if n429:
            st.info("出現限流(429)：表示並發數超過你的 Key 額度，請把左側「並發數」調低，或改用額度較高的帳號。")

    # ---------------- 分類：異常 / 通過 ----------------
    abn, normal = [], []
    for k in ok_keys:
        (abn if quick_warnings(results[k]["raw"]) else normal).append(k)

    rows = {}
    def row_of(k, state, doc, nwarn, t_txt, model, note=""):
        rows[k] = {"檔案": names[k], "狀態": state, "單號": doc, "待處理異常": str(nwarn),
                   "AI 耗時": t_txt, "模型": model, "說明": note}

    def time_model(k):
        s = results[k]["stats"]
        if s["cached"]:
            return (f"快取（原 {s['orig_api']} 秒）" if s.get("orig_api") is not None else "快取",
                    s["model"] or "不明（舊快取）")
        return f"{s['api']:.1f} 秒", s["model"]

    for k in failed:
        row_of(k, "❌ 失敗", "-", "-", "-", "-", results[k]["error"].splitlines()[0][:40])
    for k in normal:
        t, m = time_model(k)
        row_of(k, "✅ 通過檢查", results[k]["raw"].get("doc_number") or "（空）", 0, t, m)

    summary_ph = st.empty()
    exports = []

    # ---------------- ③ 需要人工處理 ----------------
    st.subheader(f"③ 需要人工處理：{len(abn) + len(failed)} 張（異常 {len(abn)}、失敗 {len(failed)}）")
    if not abn and not failed:
        st.success("🎉 全部單據都通過自動檢查，沒有需要逐張處理的。")

    if failed:
        with st.expander(f"❌ 識別失敗 {len(failed)} 張（可降低並發數後重新識別，成功的不會重跑）", expanded=False):
            for k in failed:
                st.write(f"**{names[k]}**：{results[k]['error'].splitlines()[0]}")

    for k in abn:
        data, s = results[k]["raw"], results[k]["stats"]
        n_raw = len(quick_warnings(data))
        with st.expander(f"⚠️ {names[k]}｜單號 {data.get('doc_number') or '（空）'}｜{n_raw} 項異常",
                         expanded=len(abn) <= 3):
            col_img, col_data = st.columns([1, 1.3])
            with col_img:
                st.image(Image.open(io.BytesIO(blobs[k])), use_container_width=True)
            with col_data:
                if s["cached"]:
                    st.info("⚡ 結果來自同一張圖片先前的識別（快取）。")
                else:
                    st.caption(f"⏱️ AI {s['api']:.1f} 秒｜上傳 {s['kb']} KB｜{s['model']}")
                    with st.expander("查看嘗試記錄"):
                        for ln in s["log"]:
                            st.write(ln)

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
                    st.error(f"仍有 {len(warns)} 項異常或待確認：")
                    for w in warns:
                        st.write(w)
                else:
                    st.success("✅ 修改後，格式與金額交叉驗算全部通過！")

                t, m = time_model(k)
                row_of(k, "⚠️ 異常" if warns else "✅ 已修正", doc_num or "（空）", len(warns), t, m)

                if st.checkbox("✍️ 我已核對這張單據的金額與欄位無誤", key=f"ok_{k}"):
                    exports.append({
                        "source_file": names[k], "doc_number": doc_num, "date": doc_date,
                        "customer_name": cust, "stated_total": total,
                        "verified_items": edited_df.to_dict(orient="records"),
                        "audit_log": {"verified_by": "Operator_Manual_Check",
                                      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                      "remaining_warnings_count": len(warns),
                                      "ai_original_snapshot": data}})
    for k in abn:                       # 展開區沒被渲染到時（極少數），仍保證總覽表有這一列
        if k not in rows:
            t, m = time_model(k)
            row_of(k, "⚠️ 異常", results[k]["raw"].get("doc_number") or "（空）",
                   len(quick_warnings(results[k]["raw"])), t, m)

    # ---------------- ④ 通過檢查的單據 ----------------
    st.subheader(f"④ 已通過自動檢查：{len(normal)} 張")
    if normal:
        st.caption("「通過」只代表算術一致、沒有模糊標記，不代表 AI 一定讀對，建議抽查幾張。")
        with st.expander("查看通過的單據清單與抽查", expanded=False):
            st.dataframe(pd.DataFrame([{
                "檔案": names[k], "單號": results[k]["raw"].get("doc_number", ""),
                "日期": results[k]["raw"].get("date", ""),
                "客戶": results[k]["raw"].get("customer_name", ""),
                "品項數": len(results[k]["raw"].get("items", [])),
                "總金額": results[k]["raw"].get("stated_total", 0)} for k in normal]),
                use_container_width=True, hide_index=True)
            pick = st.selectbox("抽查其中一張：", normal, format_func=lambda k: names[k])
            cc1, cc2 = st.columns([1, 1.3])
            cc1.image(Image.open(io.BytesIO(blobs[pick])), use_container_width=True)
            cc2.dataframe(pd.DataFrame(results[pick]["raw"].get("items", [])), use_container_width=True)
        if st.checkbox(f"✍️ 我已抽查，確認這 {len(normal)} 張通過檢查的單據一併匯出", key="ok_normal"):
            for k in normal:
                d = results[k]["raw"]
                exports.append({
                    "source_file": names[k], "doc_number": d.get("doc_number", ""),
                    "date": d.get("date", ""), "customer_name": d.get("customer_name", ""),
                    "stated_total": d.get("stated_total", 0.0), "verified_items": d.get("items", []),
                    "audit_log": {"verified_by": "Auto_Rule_Check_Passed + Operator_Sample_Check",
                                  "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                  "remaining_warnings_count": 0, "ai_original_snapshot": d}})

    # ---------------- 總覽表（所有單據，依上傳順序） ----------------
    summary_ph.dataframe(pd.DataFrame([rows[k] for k in blobs if k in rows]).astype(str),
                         use_container_width=True, hide_index=True)

    # ---------------- ⑤ 匯出 ----------------
    st.divider()
    st.subheader("⑤ 匯出已確認的單據")
    st.write(f"已確認 {len(exports)} / {len(ok_keys)} 張")
    if exports:
        st.download_button("💾 下載已確認單據 JSON（模擬寫入 TMS）",
                           data=json.dumps(exports, ensure_ascii=False, indent=2),
                           file_name=f"verified_batch_{datetime.now():%Y%m%d_%H%M%S}.json",
                           mime="application/json")