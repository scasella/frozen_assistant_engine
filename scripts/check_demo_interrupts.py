"""Interrupted arena runs must release the GPU lock: client disconnect, the Stop button (/api/cancel), and pre-emption
by a new run. Needs a running studio (./run_demo.sh); talks to it over HTTP.

    .venv/bin/python scripts/check_demo_interrupts.py
"""
import json, sys, threading, time
import httpx
URL = "http://127.0.0.1:8000"
T = httpx.Timeout(60.0, connect=5.0)
ok = True

def busy():
    return httpx.get(f"{URL}/api/status", timeout=5).json()["busy"]

def wait_idle(limit=15.0):
    t0 = time.time()
    while time.time() - t0 < limit:
        if not busy():
            return time.time() - t0
        time.sleep(0.25)
    return None

def report(name, passed, detail):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

def consume(body, out, stop_after=None):
    out["types"], out["tokens"] = [], 0
    with httpx.stream("POST", f"{URL}/api/stream", json=body, timeout=T) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            out["types"].append(ev["type"])
            if ev["type"] == "status":
                out.setdefault("status", []).append(ev["msg"])
            if ev["type"] == "token":
                out["tokens"] += 1
                if stop_after and out["tokens"] >= stop_after:
                    return                      # leaving the with-block closes the connection mid-stream
            if ev["type"] == "done":
                out.setdefault("done", {})[ev["col"]] = ev["text"]

# 1. client disconnect after 8 tokens
a = {}
consume({"model": "3B", "prompt_id": "t_qa_04"}, a, stop_after=8)
dt = wait_idle()
report("disconnect releases the lock", dt is not None, f"read {a['tokens']} tokens, then closed; idle after {dt if dt is None else round(dt, 2)} s")

# 2. a fresh run starts without waiting for a lock
b = {}
t0 = time.time()
consume({"model": "3B", "prompt_id": "t_ng_02", "columns": ["engine"]}, b)
waited = any("waiting" in m for m in b.get("status", []))
report("next run starts immediately", "done" in b and not waited, f"engine answered {b.get('done', {}).get('engine')!r} in {time.time() - t0:.1f} s; waited for lock: {waited}")

# 3. /api/cancel mid-stream (the UI's Stop button)
c = {}
th = threading.Thread(target=consume, args=({"model": "3B", "prompt_id": "t_qa_04"}, c))
th.start(); time.sleep(4)
httpx.post(f"{URL}/api/cancel", timeout=5)
th.join(30)
dt = wait_idle()
report("Stop (/api/cancel) ends the stream and releases the lock", not th.is_alive() and "cancelled" in c["types"] and dt is not None,
       f"{c['tokens']} tokens before cancel; cancelled event: {'cancelled' in c['types']}; idle after {dt if dt is None else round(dt, 2)} s")

# 4. pre-emption: a new run while one is streaming (clicking another preset)
d, e = {}, {}
th = threading.Thread(target=consume, args=({"model": "3B", "prompt_id": "t_qa_04"}, d))
th.start(); time.sleep(4)
t0 = time.time()
consume({"model": "3B", "prompt_id": "t_ng_02", "columns": ["engine"]}, e)
th.join(30)
report("a new run pre-empts the running one", "cancelled" in d["types"] and "done" in e and not th.is_alive(),
       f"first run cancelled after {d['tokens']} tokens; second finished in {time.time() - t0:.1f} s")
dt = wait_idle()
report("idle afterwards", dt is not None, f"busy cleared after {dt if dt is None else round(dt, 2)} s")
sys.exit(0 if ok else 1)
