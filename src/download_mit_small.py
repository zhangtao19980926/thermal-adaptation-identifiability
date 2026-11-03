# -*- coding: utf-8 -*-
import urllib.request, ssl, re, time, os
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
OUT = r"D:\thermal-adaptation-series\p1\data\raw\mit"
small = {50278161: "survey_reactionTime_uiClick.csv", 50278164: "Summary_of_derived_data.xlsx",
         50278170: "participants.csv", 50278176: "Exit_survey_and_interview_questions.pdf",
         50278185: "consolidated_atlas.csv"}

def fetch(fid, fallback, tries=40, wait=6):
    u = "https://figshare.com/ndownloader/files/%d" % fid
    for i in range(tries):
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})  # plain GET, no Range
        try:
            r = urllib.request.urlopen(req, timeout=60, context=ctx)
            st = r.status
        except urllib.error.HTTPError as e:
            st, r = e.code, e
        if st == 202:
            print(fid, "202 preparing (%d)" % i, flush=True); time.sleep(wait); continue
        if st in (200, 206):
            cd = r.headers.get("Content-Disposition", "") or ""
            m = re.search(r'filename\*?=[^;\n]*?([^"\';\n]+\.\w+)', cd)
            name = m.group(1) if m else fallback
            data = r.read()
            p = os.path.join(OUT, name)
            with open(p, "wb") as f: f.write(data)
            print("SAVED", fid, name, len(data), "bytes", flush=True); return True
        print(fid, "unexpected status", st, flush=True); time.sleep(wait)
    print(fid, "FAILED still not ready", flush=True); return False

for fid, nm in small.items():
    fetch(fid, nm)
print("DONE")
