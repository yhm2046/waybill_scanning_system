import pandas as pd
import json
import io
import time
import hashlib
import threading
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

MIN_TIMEOUT = 10      # Google 拒絕小於約 10 秒的超時設定（會回 400 deadline too short）

class RateLimiter:
    """把請求「排開」：每分鐘最多 rpm 張開始處理，避免一下子撞上額度（429）。"""
    def __init__(self, rpm):
        self.interval, self.lock, self.next = 60.0 / rpm, threading.Lock(), 0.0
    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.interval
        if t > now:
            time.sleep(t - now)

def call_with_fallback(img_bytes, api_key, models, timeout_s, budget_s, hedge_s=0):
    """備援競速：hedge_s>0 時，第一個模型超過 hedge_s 秒沒回應就同時再送下一個，誰先成功用誰；
    hedge_s=0 時不提早備援，但請求卡住超過 timeout_s 秒（SDK 的超時不一定準時生效）一樣會換下一個模型。
    總時間受 budget_s 限制。"""
    log, deadline = [], time.time() + budget_s
    stall = hedge_s if hedge_s > 0 else timeout_s     # 請求卡住超過這麼多秒，就同時換下一個模型再送
    seq = list(models) + [models[0]]        # 最後再補一路同模型（換一條後端線路，常常就不卡了）
    ex = ThreadPoolExecutor(max_workers=len(seq))
    pending, state = {}, {"i": 0, "last": 0.0}

    def launch():
        m = seq[state["i"]]
        state["i"] += 1
        rem = deadline - time.time()
        f = ex.submit(call_gemini, img_bytes, api_key, m, max(MIN_TIMEOUT, min(timeout_s, rem)))
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
            if more:
                wait_t = min(rem, max(0.1, stall - (time.time() - state["last"])))
            else:
                wait_t = rem
            done, _ = wait(list(pending), timeout=wait_t, return_when=FIRST_COMPLETED)
            for f in done:
                m, t = pending.pop(f)
                try:
                    data = f.result()
                    log.append(f"{m}：成功（{time.time() - t:.1f} 秒）")
                    return data, m, log
                except Exception as e:
                    msg = str(e)
                    log.append(f"{m}：失敗（{time.time() - t:.1f} 秒）{msg[:220]}")
                    if "API_KEY_INVALID" in msg or "API key not valid" in msg:
                        raise RuntimeError("API Key 無效，請檢查。")
                    if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and deadline - time.time() > 6:
                        time.sleep(1.5)         # 被限流：稍等一下再試，不要馬上又撞上去
            can = state["i"] < len(seq) and deadline - time.time() > MIN_TIMEOUT + 1
            if not pending and not can:
                raise RuntimeError("所有模型都失敗了，或剩餘時間不足以再試一次：\n" + "\n".join(log))
            if can and (done or time.time() - state["last"] >= stall):
                launch()
    finally:
        ex.shutdown(wait=False)             # 卡住的請求不再等，讓它自己超時結束

def extract_from_image(image, api_key, models, max_side, timeout_s, budget_s, use_cache, hedge_s=0, limiter=None):
    t0 = time.time()
    img_bytes = prepare_image(image, max_side)
    stats = {"prepare": time.time() - t0, "kb": len(img_bytes) // 1024,
             "api": 0.0, "cached": False, "model": "", "log": [], "n429": 0}

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

    if limiter is not None:
        limiter.wait()                      # 排隊等名額（不計入這張的等待上限）
    t1 = time.time()
    data, used, log = call_with_fallback(img_bytes, api_key, models, timeout_s, budget_s, hedge_s)
    stats.update({"api": time.time() - t1, "model": used, "log": log,
                  "n429": sum(1 for l in log if "429" in l or "RESOURCE_EXHAUSTED" in l)})
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
# 4. 批次處理（固定並發數的任務隊列）
# ==========================================
def run_batch(blobs, api_key, models, max_side, timeout_s, budget_s, use_cache,
              progress=None, hedge_s=0, concurrency=10, rpm=0, spread=True):
    """blobs: {key: bytes}。同時最多 concurrency 張在處理，一張完成再補下一張。
    每張各自重試/換模型；錯誤只影響那一張。hedge_s=0 表示關閉備援。
    spread=True：多個模型輪流當「第一選擇」，把請求分攤到各模型的額度上。"""
    results, t0 = {}, time.time()
    limiter = RateLimiter(rpm) if rpm and rpm > 0 else None
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(blobs)))) as ex:
        futs = {}
        for idx, (k, b) in enumerate(blobs.items()):
            img = Image.open(io.BytesIO(b))
            n = len(models)
            # 分流：第 1 張先用模型 A、第 2 張先用模型 B……每個模型各有自己的每分鐘額度，
            # 這樣兩個模型的額度可以一起用，而不是全部擠在第一個模型上
            order = models[idx % n:] + models[:idx % n] if (spread and n > 1) else models
            futs[ex.submit(extract_from_image, img, api_key, order,
                           max_side, timeout_s, budget_s, use_cache, hedge_s, limiter)] = k
        done = 0
        for f in as_completed(futs):
            k = futs[f]
            try:
                data, stats = f.result()
                results[k] = {"ok": True, "raw": data, "stats": stats}
            except Exception as e:
                results[k] = {"ok": False, "error": str(e)}
            done += 1
            if progress is not None:
                progress.progress(done / len(futs), text=f"已完成 {done}/{len(futs)} 張")
    return results, time.time() - t0

def quick_warnings(data):
    df = pd.DataFrame(data.get("items", []))
    return validate_delivery_note(data, df, float(data.get("stated_total", 0.0)))