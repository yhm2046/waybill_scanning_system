import io
import hashlib
import statistics
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from core import validate_delivery_note, run_batch, quick_warnings, test_key
from export_excel import build_excel, num

MAX_FILES = 50          # 一次最多上傳幾張
HEDGE_MAX_IMAGES = 5    # 超過這個張數，自動關閉備援競速（避免請求翻倍、更容易被限流）
ITEM_COLS = ["item_name", "quantity", "unit_price", "subtotal", "is_unclear"]

st.set_page_config(page_title="LOGFLOWS 單據識別與驗證 Demo", layout="wide")
st.title("📦 物流單據 AI 批次識別與防呆驗證 Demo")
st.caption("上傳 ➔ 並發識別 ➔ 規則自動檢查 ➔ 只有異常的單據需要人工處理 ➔ 匯出 Excel")


# ---------------- 小工具 ----------------
def make_thumb(b: bytes, max_side: int = 900) -> bytes:
    """顯示用縮圖：頁面每次重新整理都要畫圖，用小圖比用原圖快很多。"""
    im = Image.open(io.BytesIO(b)).convert("RGB")
    im.thumbnail((max_side, max_side))
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=80)
    return out.getvalue()


def items_df(items) -> pd.DataFrame:
    """固定欄位；即使 AI 沒讀到任何明細，表格也能讓人手動新增一行。"""
    df = pd.DataFrame(items or []).reindex(columns=ITEM_COLS)
    # 明確指定每欄的資料型態：沒有明細（或 AI 少給某欄）時欄位會是 float/空值，
    # 勾選框欄位必須是 bool，否則 Streamlit 會報 checkbox 與 float 不相容
    df["item_name"] = df["item_name"].fillna("").astype(str)
    for c in ("quantity", "unit_price", "subtotal"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["is_unclear"] = df["is_unclear"].fillna(False).astype(bool)
    return df


def norm_items(df_or_list):
    """把明細整理成可比較的 list（用來判斷人工有沒有改過）。"""
    rows = df_or_list.to_dict(orient="records") if isinstance(df_or_list, pd.DataFrame) else (df_or_list or [])
    out = []
    for r in rows:
        name = r.get("item_name")
        out.append({
            "item_name": "" if name is None or (isinstance(name, float) and name != name) else str(name).strip(),
            "quantity": round(num(r.get("quantity")), 2),
            "unit_price": round(num(r.get("unit_price")), 2),
            "subtotal": round(num(r.get("subtotal")), 2),
            "is_unclear": bool(r.get("is_unclear")) if r.get("is_unclear") == r.get("is_unclear") else False,
        })
    return out


def time_model(results, k):
    s = results[k]["stats"]
    if s["cached"]:
        t = f"快取（原 {s['orig_api']} 秒）" if s.get("orig_api") is not None else "快取"
        return t, s["model"] or "不明（舊快取）"
    return f"{s['api']:.1f} 秒", s["model"]


# ---------------- 側邊欄 ----------------
with st.sidebar:
    st.header("⚙️ 運行設定")
    api_key = st.text_input("輸入 Gemini API Key：", type="password", help="可於 Google AI Studio 取得")
    api_key = api_key.strip().strip('"').strip("'")
    models_text = st.text_input(
        "模型名稱（可填多個，逗號隔開）",
        value="gemini-3.1-flash-lite, gemini-3.5-flash-lite",
        help="多個模型會輪流分攤請求，並在失敗時互相備援。名稱請以「測試 Key」列出的為準")
    models = [m.strip() for m in models_text.split(",") if m.strip()]
    operator = st.text_input("核對人員（寫入 Excel）", value="Operator")

    concurrency = st.number_input("並發數（同時處理幾張）", 1, 50, 10, 1,
                                  help="免費額度建議 10；太大會變慢或被限流（429）")
    max_side = st.slider("圖片最長邊（像素）", 800, 1600, 1120, 160, help="越小越快，但字太小可能讀錯")
    hedge_on = st.checkbox(f"小批量時開啟備援競速（超過 {HEDGE_MAX_IMAGES} 張自動關閉）", value=True)
    hedge_s = st.number_input("備援等待（秒）", 3, 30, 8, 1)
    timeout_s = st.number_input("單次請求超時（秒）", 10, 300, 15, 1,
                                help="正常約 5 秒；卡住超過這個時間，會自動換另一個模型重試")
    rpm = st.number_input("每分鐘最多送出幾張（0=不限）", 0, 600, 0, 5,
                          help="被限流（429）時填你的 Key 額度，程式會自動排開請求")
    budget_s = st.number_input("每張總等待上限（秒）", 20, 600, 30, 5,
                               help="免費額度較慢時，可放寬到 45，成功率更高")
    use_cache = st.checkbox("同一張圖片使用上次識別結果（快取）", value=True,
                            help="做速度測試時請取消勾選，否則快取的圖片不會真的呼叫 AI")

    if st.button("🔑 測試 Key", use_container_width=True):
        with st.spinner("測試中…"):
            ok, msg = test_key(api_key, models)
        (st.success if ok else st.error)(msg)

    st.divider()
    st.markdown("**💡 設計原則：**\n1. 固定並發數的隊列，請求分攤到多個模型\n2. 某一張失敗不影響其他張\n"
                "3. 算術與格式由 Python 固定規則把關\n4. 只有異常、失敗的單據才需要人工逐張核對\n"
                "5. 結果匯出 Excel，保留人工修改與核對紀錄")

# ---------------- ① 上傳 ----------------
uploaded = st.file_uploader(f"① 請上傳送貨單或發票圖片（最多 {MAX_FILES} 張，JPG / PNG）",
                            type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if len(uploaded) > MAX_FILES:
    st.warning(f"一次最多 {MAX_FILES} 張，只取前 {MAX_FILES} 張。")
    uploaded = uploaded[:MAX_FILES]

if uploaded:
    n_run = int(st.number_input("本次處理前幾張（建議先測 10，再測 20，最後 50）",
                                1, len(uploaded), min(10, len(uploaded)), 1))
    uploaded = uploaded[:n_run]

blobs = {}
for i, f in enumerate(uploaded):
    b = f.getvalue()
    blobs[f"{i}_{hashlib.md5(b).hexdigest()[:8]}"] = b
names = {k: uploaded[i].name for i, k in enumerate(blobs)}

eff_hedge = hedge_s if (hedge_on and len(blobs) <= HEDGE_MAX_IMAGES) else 0
eff_conc = min(int(concurrency), max(1, len(blobs)))

# ---------------- ② 識別 ----------------
if blobs:
    st.caption(f"本次設定：處理 {len(blobs)} 張｜並發 {eff_conc}｜備援 "
               f"{'關閉' if eff_hedge == 0 else f'{eff_hedge} 秒'}｜快取 {'開' if use_cache else '關'}")
    if st.button(f"🔍 ② 開始識別這 {len(blobs)} 張", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="已送出，等待結果…")
        results, wall = run_batch(blobs, api_key, models, max_side, timeout_s, budget_s, use_cache,
                                  bar, eff_hedge, eff_conc, int(rpm))
        bar.empty()
        st.session_state["batch"] = {
            "results": results, "wall": wall, "keys": list(blobs),
            "conc": eff_conc, "hedge": eff_hedge, "models": models,
            # 只算一次，之後每次點選畫面都不必重算
            "thumbs": {k: make_thumb(b) for k, b in blobs.items()},
            "warn_n": {k: len(quick_warnings(r["raw"])) for k, r in results.items() if r["ok"]},
        }

batch = st.session_state.get("batch")
if blobs and batch and set(batch["keys"]) == set(blobs):
    results, wall = batch["results"], batch["wall"]
    thumbs, warn_n = batch["thumbs"], batch["warn_n"]

    # ---------------- 統計 ----------------
    ok_keys = [k for k in blobs if results[k]["ok"]]
    fresh = [k for k in ok_keys if not results[k]["stats"]["cached"]]
    cached = [k for k in ok_keys if results[k]["stats"]["cached"]]
    failed = [k for k in blobs if not results[k]["ok"]]
    abn = [k for k in ok_keys if warn_n[k] > 0]
    normal = [k for k in ok_keys if warn_n[k] == 0]
    n429 = sum(results[k]["stats"].get("n429", 0) for k in ok_keys) + \
        sum(results[k]["error"].count("429") for k in failed)
    lat = sorted(results[k]["stats"]["api"] for k in fresh)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("總張數", len(blobs))
    m2.metric("🆕 新識別", len(fresh))
    m3.metric("⚡ 快取", len(cached))
    m4.metric("⚠️ 需人工處理", len(abn) + len(failed))
    m5.metric("✅ 通過檢查", len(normal))

    if len(failed) == len(blobs):
        st.error("全部失敗，請先按左側「🔑 測試 Key」。")
    elif not fresh:
        st.warning("這次沒有真正呼叫 AI，結果都來自舊記錄（速度測試請取消快取）。")
    else:
        detail = f"⏱️ 總耗時 {wall:.1f} 秒"
        if lat:
            detail += f"｜單張中位數 {statistics.median(lat):.1f} 秒、最慢 {lat[-1]:.1f} 秒"
        detail += f"｜限流(429) {n429} 次｜並發 {batch['conc']}｜備援 {'關' if batch['hedge'] == 0 else '開'}"
        good = wall <= 30 and not failed
        (st.success if good else st.warning)(detail + ("｜✅ 30 秒內全部完成" if good else "｜❌ 沒有全部在 30 秒內完成"))
        if n429:
            st.info("出現限流(429)：請把左側「並發數」調低，或填入「每分鐘最多送出幾張」，或改用額度較高的帳號。")

    # 逐張記錄：總覽表與匯出都用這份
    rows = {}
    entries = {}        # k -> 要寫進 Excel 的單據資料（含是否已確認）

    def row_of(k, state, doc, nwarn, t_txt, model, note=""):
        rows[k] = {"檔案": names[k], "狀態": state, "單號": doc, "待處理異常": str(nwarn),
                   "AI 耗時": t_txt, "模型": model, "說明": note}

    for k in failed:
        row_of(k, "❌ 失敗", "-", "-", "-", "-", results[k]["error"].splitlines()[0][:40])

    summary_ph = st.empty()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------- ③ 需要人工處理 ----------------
    st.subheader(f"③ 需要人工處理：{len(abn) + len(failed)} 張（異常 {len(abn)}、失敗 {len(failed)}）")
    if not abn and not failed:
        st.success("🎉 全部單據都通過自動檢查，沒有需要逐張處理的。")

    if failed:
        with st.expander(f"❌ 識別失敗 {len(failed)} 張（可降低並發數後重新識別，成功的不會重跑）"):
            for k in failed:
                st.write(f"**{names[k]}**：{results[k]['error'].splitlines()[0]}")

    for k in abn:
        data, s = results[k]["raw"], results[k]["stats"]
        with st.expander(f"⚠️ {names[k]}｜單號 {data.get('doc_number') or '（空）'}｜{warn_n[k]} 項異常",
                         expanded=len(abn) <= 3):
            col_img, col_data = st.columns([1, 1.3])
            with col_img:
                st.image(thumbs[k], use_container_width=True)
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
                    items_df(data.get("items")),
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

                confirmed = st.checkbox("✍️ 我已核對這張單據的金額與欄位無誤", key=f"ok_{k}")

                items = norm_items(edited_df)
                modified = (
                    items != norm_items(data.get("items"))
                    or doc_num.strip() != str(data.get("doc_number", "")).strip()
                    or doc_date.strip() != str(data.get("date", "")).strip()
                    or cust.strip() != str(data.get("customer_name", "")).strip()
                    or round(total, 2) != round(num(data.get("stated_total")), 2))
                t, m = time_model(results, k)
                row_of(k, "✍️ 已確認" if confirmed else ("⚠️ 異常" if warns else "✅ 已修正"),
                       doc_num or "（空）", len(warns), t, m)
                entries[k] = {
                    "source_file": names[k], "doc_number": doc_num, "date": doc_date,
                    "customer_name": cust, "stated_total": total, "items": items,
                    "status": ("人工確認（已修改）" if modified else "人工確認") if confirmed else "未確認（有異常）",
                    "modified": modified, "warnings": warns, "model": m,
                    "verified_by": operator if confirmed else "", "timestamp": now_str if confirmed else "",
                    "confirmed": confirmed, "todo_reason": f"尚未確認，仍有 {len(warns)} 項異常"}

    # ---------------- ④ 通過檢查的單據 ----------------
    st.subheader(f"④ 已通過自動檢查：{len(normal)} 張")
    ok_normal = False
    if normal:
        st.caption("「通過」只代表算術一致、沒有模糊標記，不代表 AI 一定讀對，建議抽查幾張。")
        with st.expander(f"查看全部 {len(normal)} 張通過的單據（清單＋逐張原圖）"):
            st.dataframe(pd.DataFrame([{
                "檔案": names[k], "單號": results[k]["raw"].get("doc_number", ""),
                "日期": results[k]["raw"].get("date", ""),
                "客戶": results[k]["raw"].get("customer_name", ""),
                "品項數": len(results[k]["raw"].get("items", [])),
                "總金額": results[k]["raw"].get("stated_total", 0)} for k in normal]),
                use_container_width=True, hide_index=True)

            # 每一張通過的單據都可以展開看原圖與識別結果（Streamlit 不允許在 expander 裡再放 expander，
            # 所以這裡用「單選按鈕 + 上一張/下一張」逐張瀏覽，並另外提供「一次全部顯示」）
            view_mode = st.radio("查看方式", ["逐張瀏覽", "一次全部顯示"], horizontal=True, key="normal_view")

            def show_doc(k):
                d = results[k]["raw"]
                cc1, cc2 = st.columns([1, 1.3])
                cc1.image(thumbs[k], use_container_width=True)
                cc2.write(f"**{names[k]}**｜單號 {d.get('doc_number') or '（空）'}｜日期 {d.get('date', '')}")
                cc2.write(f"客戶：{d.get('customer_name', '')}｜單據總金額：{d.get('stated_total', 0)}")
                cc2.dataframe(items_df(d.get("items")), use_container_width=True)

            if view_mode == "逐張瀏覽":
                if st.session_state.get("normal_idx", 0) >= len(normal):
                    st.session_state["normal_idx"] = 0
                nav1, nav2, nav3 = st.columns([1, 1, 4])
                if nav1.button("⬅️ 上一張", use_container_width=True, key="normal_prev"):
                    st.session_state["normal_idx"] = (st.session_state.get("normal_idx", 0) - 1) % len(normal)
                if nav2.button("下一張 ➡️", use_container_width=True, key="normal_next"):
                    st.session_state["normal_idx"] = (st.session_state.get("normal_idx", 0) + 1) % len(normal)
                idx = st.session_state.get("normal_idx", 0)
                nav3.markdown(f"第 **{idx + 1} / {len(normal)}** 張")
                show_doc(normal[idx])
            else:
                if len(normal) > 20:
                    st.caption("單據較多，頁面會比較長、載入較慢。")
                for n_, k in enumerate(normal, 1):
                    st.markdown(f"##### {n_}. {names[k]}")
                    show_doc(k)
                    st.divider()
        ok_normal = st.checkbox(f"✍️ 我已抽查，確認這 {len(normal)} 張通過檢查的單據無誤", key="ok_normal")

    for k in normal:
        d = results[k]["raw"]
        t, m = time_model(results, k)
        row_of(k, "✅ 通過檢查", d.get("doc_number") or "（空）", 0, t, m)
        entries[k] = {
            "source_file": names[k], "doc_number": d.get("doc_number", ""), "date": d.get("date", ""),
            "customer_name": d.get("customer_name", ""), "stated_total": d.get("stated_total", 0.0),
            "items": norm_items(d.get("items")),
            "status": "通過檢查·抽查確認" if ok_normal else "通過檢查·未確認",
            "modified": False, "warnings": [], "model": m,
            "verified_by": operator if ok_normal else "", "timestamp": now_str if ok_normal else "",
            "confirmed": ok_normal, "todo_reason": "通過自動檢查，但尚未抽查確認"}

    # ---------------- 總覽表（所有單據，依上傳順序） ----------------
    summary_ph.dataframe(pd.DataFrame([rows[k] for k in blobs if k in rows]).astype(str),
                         use_container_width=True, hide_index=True)

    # ---------------- ⑤ 匯出 Excel ----------------
    st.divider()
    st.subheader("⑤ 匯出 Excel")
    scope = st.radio("匯出範圍", ["只匯出已確認的單據", "匯出全部成功識別的單據（未確認的會標示狀態）"],
                     horizontal=True)
    only_confirmed = scope.startswith("只")

    ordered = [entries[k] for k in blobs if k in entries]
    export_docs = [e for e in ordered if e["confirmed"]] if only_confirmed else ordered
    todo_rows = [{"source_file": names[k], "status": "識別失敗", "reason": results[k]["error"].splitlines()[0]}
                 for k in failed]
    if only_confirmed:
        todo_rows += [{"source_file": e["source_file"], "status": "未確認", "reason": e["todo_reason"]}
                      for e in ordered if not e["confirmed"]]

    st.write(f"已確認 {sum(e['confirmed'] for e in ordered)} / {len(ok_keys)} 張｜"
             f"本次將匯出 {len(export_docs)} 張｜未處理 {len(todo_rows)} 張（會列在「未處理單據」工作表）")

    if export_docs:
        mc = Counter(results[k]["stats"]["model"] for k in ok_keys if results[k]["stats"]["model"])
        meta = {
            "匯出時間": now_str, "核對人員": operator, "匯出範圍": scope,
            "批次張數": len(blobs), "識別成功": len(ok_keys), "識別失敗": len(failed),
            "新識別／快取": f"{len(fresh)} ／ {len(cached)}", "本次識別總耗時（秒）": round(wall, 1),
            "並發數": batch["conc"], "模型分佈": "，".join(f"{m}：{n} 張" for m, n in mc.items()) or "（快取）",
            "說明": "AI 識別結果須經人工核對；金額欄位以 Excel 公式重新計算，可與單據標示總額交叉驗證。"}
        xlsx = build_excel(export_docs, todo_rows, meta)
        st.download_button("📥 下載 Excel（.xlsx）", data=xlsx,
                           file_name=f"單據識別結果_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary")
    else:
        st.info("還沒有已確認的單據。請在上方勾選「我已核對…」，或改選「匯出全部成功識別的單據」。")