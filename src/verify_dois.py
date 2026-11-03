# -*- coding: utf-8 -*-
"""Verify every DOI in refs_library against the Crossref REST API."""
import sys, re, time, json, urllib.request, urllib.parse, urllib.error, os
sys.path.insert(0, r"D:\thermal-adaptation-series\p1\src\p1mk")
from refs_library import REFS

OUT = r"D:\thermal-adaptation-series\p1\outputs\refs_verified_round2.csv"
MAILTO = "liruiqi1@gxcme.edu.cn"

def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi) + "?mailto=" + MAILTO
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0 (mailto:%s)" % MAILTO})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode("utf-8"))["message"]

rows = []
ok = fail = nodoi = 0
for k, c in REFS.items():
    m = re.search(r"DOI:\s*(\S+?)\.?\s*$", c.strip())
    doi = m.group(1).rstrip(".") if m else ""
    if not doi:
        rows.append((k, "", "NO_DOI", "", "", "", c[:80])); nodoi += 1
        print("[no-DOI] %s" % k); continue
    try:
        msg = crossref(doi)
        title = (msg.get("title") or [""])[0][:70]
        cont = (msg.get("container-title") or [""])[0][:40]
        yr = (msg.get("issued", {}).get("date-parts", [[None]])[0][0])
        rows.append((k, doi, "OK", str(yr), cont, title, ""))
        ok += 1; print("[OK]   %-22s %s  %s %s" % (k, doi, yr, cont))
    except urllib.error.HTTPError as e:
        rows.append((k, doi, "HTTP_%s" % e.code, "", "", "", c[:80]))
        fail += 1; print("[FAIL] %-22s %s  HTTP %s" % (k, doi, e.code))
    except Exception as e:
        rows.append((k, doi, "ERR:%s" % type(e).__name__, "", "", "", c[:80]))
        fail += 1; print("[FAIL] %-22s %s  %r" % (k, doi, e))
    time.sleep(0.35)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("key,doi,status,year,container,title,note\n")
    for r in rows:
        f.write(",".join('"%s"' % str(x).replace('"', "'") for x in r) + "\n")
print("\n=== DOI verify: OK=%d  FAIL=%d  NO_DOI=%d  total=%d ===" % (ok, fail, nodoi, len(REFS)))
print("written:", OUT)
