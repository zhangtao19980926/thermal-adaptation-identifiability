# -*- coding: utf-8 -*-
import urllib.request, ssl, re, os, time
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
B = "https://physionet.org/files/in-gauge-and-en-gage/1.0.0/"
ROOT = r"D:\thermal-adaptation-series\p1\data\raw\ingauge"
hdr = {"User-Agent": "Mozilla/5.0"}
SUBS = ["longitudinal/", "survey/", "participant_class_info/"]

def get(url, binary=False):
    req = urllib.request.Request(url, headers=hdr)
    raw = urllib.request.urlopen(req, timeout=90, context=ctx).read()
    return raw if binary else raw.decode("utf-8", "ignore")

def list_files(sub):
    html = get(B + sub)
    out = []
    for m in re.finditer(r'href="([^"?#]+\.\w+)">[^<]*</a>\s+[\dA-Za-z\-]+\s[\d:]+\s+(\d+)', html):
        out.append((m.group(1), int(m.group(2))))
    return out

def dl(url, path, size, tries=60):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = os.path.getsize(path) if os.path.exists(path) else 0
    if done == size and size > 0:
        print("skip", os.path.basename(path), "complete"); return
    for a in range(tries):
        try:
            h = dict(hdr)
            mode = "wb"; start = 0
            if done > 0:
                h["Range"] = "bytes=%d-" % done
            req = urllib.request.Request(url, headers=h)
            r = urllib.request.urlopen(req, timeout=90, context=ctx)
            if r.status == 206:
                mode = "ab"; start = done
            elif r.status == 200:
                mode = "wb"; done = 0; start = 0
            with open(path, mode) as f:
                while True:
                    ch = r.read(262144)
                    if not ch: break
                    f.write(ch); done += len(ch)
                    if size and done % (4 * 1024 * 1024) < 262144:
                        print("  %.0f%% (%d/%d)" % (100 * done / size, done, size), flush=True)
            if size == 0 or done >= size:
                print("OK", os.path.basename(path), done, "bytes", flush=True); return
        except Exception as e:
            print("  retry", a, os.path.basename(path), repr(e)[:80], flush=True); time.sleep(3)
    print("FAIL", path, flush=True)

grand = 0
for sub in SUBS:
    try:
        files = list_files(sub)
    except Exception as e:
        print("list fail", sub, repr(e)[:100]); continue
    print("### %s : %d files" % (sub, len(files)))
    for name, size in files:
        print("  %-40s %.2f MB" % (name, size / 1e6))
        grand += size
        dl(B + sub + name, os.path.join(ROOT, sub.replace("/", os.sep), name), size)
print("GRAND TOTAL %.1f MB" % (grand / 1e6), flush=True)
print("ALL DONE", flush=True)
