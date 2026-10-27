import streamlit as st
import pandas as pd
import json
import io
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

    calculated_Grand_total = 0.0
    for idx, row in edited_df.iterrows():
        qty = float(row.get("quantity", 0))
        price = float(row.get("unit_price", 0.0))
        sub = float(row.get("subtotal", 0.0))
        expected_sub = round(qty * price, 2)
        calculated_Grand_total += sub
        
        if abs(expected_sub - sub) > 0.01:
            warnings.append(
                f"⚠️ 【明細算術不符】第 {idx + 1} 行「{row.get('item_name')}」："
                f"數量({qty}) × 單價({price}) 應為 {expected_sub}，但單據小計為 {sub}。"
            )
        
        if row.get("is_unclear", False):
            warnings.append(
                f"👁️ 【手寫/字跡模糊】第 {idx + 1} 行「{row.get('item_name')}」包含手寫塗改或模糊字跡，請對照左側原圖確認。"
            )

    if abs(round(calculated_Grand_total, 2) - round(stated_total, 2)) > 0.01:
        warnings.append(
            f"💰 【總金額異常】明細加總金額為 ${calculated_Grand_total:.2f}，"
            f"與單據標示總金額 ${stated_total:.2f} 不一致！請人工核對。"
        )

    return warnings

# ==========================================
# 3. AI 識別層 (極速壓縮與優化)
# ==========================================
def convert_image_to_fast_bytes(image: Image.Image, max_size=800) -> bytes:
    """將圖片縮小並轉換為輕量化 JPEG 位元組流，秒速上傳"""
    image = image.convert("RGB")
    ratio = min(max_size / image.width, max_size / image.height)
    if ratio < 1.0:
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    return img_byte_arr.getvalue()

def extract_from_image(image: Image.Image, api_key: str) -> dict:
    from google import genai
    from google.genai import types
    
    # 取得極速壓縮後的 JPEG 位元組
    image_bytes = convert_image_to_fast_bytes(image)
    
    client = genai.Client(api_key=api_key)
    prompt = (
        "你是一個專業的物流單據識別助手。請讀取這張送貨單或發票圖片，"
        "嚴格按照指定的 JSON Schema 提取單號、日期、客戶名稱、貨品明細及總金額。"
        "若某行有手寫塗改或字跡模糊，請將 is_unclear 設為 true。"
    )
    
    # 使用 PIL Image 從 bytes 重新載入以符合 SDK 傳遞格式
    fast_img = Image.open(io.BytesIO(image_bytes))

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[fast_img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DeliveryNote,
            temperature=0.1,
        ),
    )
    return json.loads(response.text)

# ==========================================
# 4. Streamlit 網頁互動介面
# ==========================================
st.set_page_config(page_title="LOGFLOWS 單據識別與驗證 Demo", layout="wide")
st.title("📦 物流單據 AI 識別與防呆驗證 Demo")
st.caption("對應面談簡報架構：上傳單據 ➔ AI 極速提取 ➔ 固定規則自動檢查 ➔ 人工確認與匯出")

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 運行設定")
    api_key = st.text_input("輸入 Gemini API Key：", type="password", help="可於 Google AI Studio 免費取得")
    
    st.divider()
    st.markdown("**💡 速度優化說明：**\n1. 圖片自動進行 800px 壓縮與 JPEG 最佳化\n2. 傳輸體積減少 70% 以上，確保 15-25 秒內回應\n3. 算術與格式由 Python 固定規則把關")

# 步驟 1：上傳圖片
uploaded_file = st.file_uploader("① 請上傳送貨單或發票圖片 (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col_img, col_data = st.columns([1, 1.3])
    
    with col_img:
        st.subheader("📄 原始單據影像")
        st.image(img, use_container_width=True)
        
        if st.button("🔍 ② 執行 AI 極速讀取與檢查", type="primary", use_container_width=True):
            if not api_key:
                st.error("請先在左側輸入你的 Gemini API Key！")
            else:
                with st.spinner("AI 正在高速辨識單據中（預計 10-20 秒內完成）..."):
                    try:
                        raw_result = extract_from_image(img, api_key)
                        st.session_state["ai_raw_output"] = raw_result
                        st.session_state["parsed_data"] = json.loads(json.dumps(raw_result))
                        st.success("辨識完成！")
                    except Exception as e:
                        st.error(f"辨識失敗，請檢查網路或 API Key 是否正確：{e}")

    # 步驟 2 ~ 4：顯示表格、自動檢查、人工修改與匯出
    with col_data:
        if "parsed_data" in st.session_state:
            data = st.session_state["parsed_data"]
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
                    "is_unclear": st.column_config.CheckboxColumn("手寫/模糊待確認?")
                },
                num_rows="dynamic",
                use_container_width=True
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
                        "ai_original_snapshot": st.session_state["ai_raw_output"]
                    }
                }
                st.success("🎉 資料已成功打包！")
                st.json(final_payload)
                
                json_str = json.dumps(final_payload, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 下載 JSON 檔案",
                    data=json_str,
                    file_name=f"verified_{doc_num}.json",
                    mime="application/json"
                )