"""速度測試：不開網頁，直接測 N 張圖片的總耗時。
用法（PowerShell）：
  $env:GEMINI_API_KEY="你的Key"
  python bench.py 圖片資料夾 --n 10 --concurrency 10
  python bench.py 圖片資料夾 --n 20 --concurrency 20
加 --mock 可不連網路，用模擬的 API 檢查程式邏輯（--mock-rpm 模擬每分鐘額度）。
"""
import argparse, collections, os, statistics, sys, threading, time
from pathlib import Path
import core

ap = argparse.ArgumentParser()
ap.add_argument("folder", nargs="?", default=".")
ap.add_argument("--n", type=int, default=10, help="測試張數")
ap.add_argument("--concurrency", type=int, default=10, help="同時處理張數")
ap.add_argument("--hedge", type=float, default=0, help="備援等待秒數，0=關閉（大批量建議關閉）")
ap.add_argument("--models", default="gemini-3.1-flash-lite, gemini-3.5-flash-lite")
ap.add_argument("--max-side", type=int, default=1120)
ap.add_argument("--timeout", type=int, default=15, help="單次請求超時（秒），最小 10")
ap.add_argument("--rpm", type=int, default=0, help="每分鐘最多送出幾張，0=不限（被 429 時用來限速）")
ap.add_argument("--budget", type=int, default=30, help="每張總等待上限（秒）")
ap.add_argument("--no-spread", action="store_true", help="不把請求分攤到多個模型（預設會輪流分攤）")
ap.add_argument("--mock", action="store_true", help="用模擬 API，不連網")
ap.add_argument("--mock-latency", type=float, default=5.0)
ap.add_argument("--mock-rpm", type=int, default=1000, help="模擬每分鐘最多接受幾個請求")
a = ap.parse_args()

if a.mock:
    lock, stamps = threading.Lock(), collections.defaultdict(collections.deque)
    def fake(img_bytes, api_key, model, timeout_s):
        with lock:                      # 每個模型各自計算每分鐘額度
            now = time.time()
            q = stamps[model]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= a.mock_rpm:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            q.append(now)
        time.sleep(a.mock_latency)
        return {"doc_number": "X", "date": "2026-10-28", "customer_name": "C", "items": [], "stated_total": 0}
    core.call_gemini = fake
    key = "mock"
else:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("請先設定環境變數 GEMINI_API_KEY")

folder = Path(a.folder)
files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")] if folder.is_dir() else []
if not files and not a.mock:
    sys.exit(f"資料夾沒有圖片：{folder}")
if a.mock and not files:
    from PIL import Image
    import io
    files = None
    blobs = {}
    for i in range(a.n):
        b = io.BytesIO(); Image.new("RGB", (1600, 1200), (i * 5 % 255, 50, 50)).save(b, "JPEG"); blobs[f"m{i}"] = b.getvalue()
else:
    if len(files) < a.n:
        print(f"⚠️ 資料夾只有 {len(files)} 張，少於 {a.n} 張；為了不被快取影響，只測 {len(files)} 張。")
    blobs = {f"{i}_{p.name}": p.read_bytes() for i, p in enumerate(files[:a.n])}

models = [m.strip() for m in a.models.split(",") if m.strip()]
print(f"測試 {len(blobs)} 張｜並發 {a.concurrency}｜備援 {'關' if a.hedge <= 0 else str(a.hedge) + ' 秒'}｜模型 {models}")
results, wall = core.run_batch(blobs, key, models, a.max_side, a.timeout, a.budget,
                               use_cache=False, progress=None, hedge_s=a.hedge, concurrency=a.concurrency, rpm=a.rpm,
                               spread=not a.no_spread)

ok = [r for r in results.values() if r["ok"]]
bad = [r for r in results.values() if not r["ok"]]
lat = sorted(r["stats"]["api"] for r in ok)
n429 = sum(r["stats"]["n429"] for r in ok) + sum(r["error"].count("429") for r in bad)
print("-" * 50)
print(f"成功 {len(ok)} 張｜失敗 {len(bad)} 張｜遇到限流(429) {n429} 次")
if lat:
    print(f"單張耗時：中位數 {statistics.median(lat):.1f} 秒｜最慢 {lat[-1]:.1f} 秒")
mc = collections.Counter(r["stats"]["model"] for r in ok)
print("各模型完成張數：", dict(mc))
print(f"總耗時：{wall:.1f} 秒  →  {'✅ 30 秒內完成' if wall <= 30 and not bad else '❌ 沒有全部在 30 秒內完成'}")
print("-" * 50)
print("逐張明細：")
for k, r in results.items():
    if r["ok"]:
        st_ = r["stats"]
        print(f"  ✅ {k[:28]:28s} {st_['api']:5.1f} 秒  {st_['model']}  (429:{st_['n429']})")
    else:
        print(f"  ❌ {k[:28]:28s} 失敗")
        for ln in r["error"].splitlines()[:8]:
            print("       ", ln[:300])