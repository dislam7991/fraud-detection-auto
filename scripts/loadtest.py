import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import pandas as pd


import requests

URL = "http://localhost:8000/predict"

row = pd.read_parquet("data/splits/test.parquet").iloc[0]
BODY = {"features": json.loads(row.to_json())}

TOTAL = 2000
THREADS = 16

first = requests.post(URL, json=BODY)
print("First reply:", first.status_code, first.text)
if first.status_code != 200:
    raise SystemExit("The request failed. Fix the request body, then run again.")

local = threading.local()

def one_request(_):
    if not hasattr(local, "session"):
        local.session = requests.Session()
    start = time.perf_counter()
    reply = local.session.post(URL, json=BODY)
    return time.perf_counter() - start, reply.status_code

begin = time.perf_counter()
with ThreadPoolExecutor(max_workers=THREADS) as pool:
    results = list(pool.map(one_request, range(TOTAL)))
elapsed = time.perf_counter() - begin

times = sorted(r[0] * 1000 for r in results)
errors = sum(1 for r in results if r[1] != 200)
print(f"Requests per second: {TOTAL / elapsed:.0f}")
print(f"Median latency: {statistics.median(times):.1f} ms")
print(f"95th percentile latency: {times[int(len(times) * 0.95)]:.1f} ms")
print(f"Errors: {errors} of {TOTAL}")