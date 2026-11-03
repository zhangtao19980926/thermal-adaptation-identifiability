# -*- coding: utf-8 -*-
"""Generic resumable downloader (HTTP Range), usage: python download_url.py URL OUTFILE"""
import os, sys, urllib.request, ssl, time
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE

def robust_download(url, path, max_tries=200, chunk=1 << 18):
    for attempt in range(max_tries):
        done = os.path.getsize(path) if os.path.exists(path) else 0
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        total = None
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                if done and r.status == 206:
                    cr = r.headers.get("Content-Range", "")
                    total = int(cr.split("/")[-1]) if "/" in cr and cr.split("/")[-1].isdigit() else None
                    mode = "ab"
                elif r.status == 200:
                    total = int(r.headers.get("Content-Length", 0)) or None
                    done = 0; mode = "wb"
                else:
                    print("status", r.status, flush=True); time.sleep(2); continue
                with open(path, mode) as f:
                    while True:
                        b = r.read(chunk)
                        if not b: break
                        f.write(b); done += len(b)
                if total is None or done >= total:
                    print(f"DONE {done} bytes -> {path}", flush=True); return True
                print(f"  {done}/{total} ({100*done/total:.1f}%) attempt {attempt+1}", flush=True)
        except Exception as e:
            print(f"  attempt {attempt+1} at {done}B: {repr(e)[:90]}", flush=True)
            time.sleep(1.5)
    return False

if __name__ == "__main__":
    url, out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ok = robust_download(url, out)
    print("SUCCESS" if ok else "FAILED", flush=True)
