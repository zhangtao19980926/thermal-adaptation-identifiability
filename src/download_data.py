# -*- coding: utf-8 -*-
"""Download missing public longitudinal datasets (stdlib only)."""
import urllib.request, zipfile, io, os, json, sys

RAW = r'D:\thermal-adaptation-series\p1\data\raw'
os.makedirs(RAW, exist_ok=True)

def get(url, timeout=120):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return urllib.request.urlopen(req, timeout=timeout).read()

# ---- 1. DORN longitudinal study (resumable direct download) ----
def robust_download(url, path, max_tries=40, chunk=1 << 16):
    """Download with HTTP Range resume; survives repeated connection drops."""
    for attempt in range(max_tries):
        done = os.path.getsize(path) if os.path.exists(path) else 0
        headers = {'User-Agent': 'Mozilla/5.0'}
        total = None
        if done:
            headers['Range'] = f'bytes={done}-'
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                if done and r.status == 206:
                    cr = r.headers.get('Content-Range', '')
                    total = int(cr.split('/')[-1]) if '/' in cr else None
                    mode = 'ab'
                elif r.status == 200:
                    total = int(r.headers.get('Content-Length', 0)) or None
                    done = 0; mode = 'wb'
                else:
                    print('  unexpected status', r.status, flush=True); continue
                with open(path, mode) as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b); done += len(b)
            if total is None or done >= total:
                print(f'  completed {done} bytes', flush=True)
                return True
            print(f'  partial {done}/{total} (attempt {attempt+1}), resuming', flush=True)
        except Exception as e:
            print(f'  attempt {attempt+1} at {done}B: {repr(e)[:100]}', flush=True)
    return False

ok = False
for branch in ('main', 'master'):
    url = (f'https://github.com/FedericoTartarini/dorn-longitudinal-tc-study/'
           f'archive/refs/heads/{branch}.zip')
    zippath = os.path.join(RAW, f'dorn_{branch}.zip')
    if os.path.exists(zippath):
        os.remove(zippath)
    print('resumable download DORN', branch, flush=True)
    if robust_download(url, zippath):
        try:
            with zipfile.ZipFile(zippath) as z:
                outdir = os.path.join(RAW, 'dorn')
                z.extractall(outdir)
            print('extracted to', outdir, flush=True)
            ok = True
            break
        except Exception as e:
            print('  zip corrupt, retrying branch:', repr(e)[:120], flush=True)
            os.remove(zippath)
if not ok:
    print('DORN DOWNLOAD FAILED', flush=True)

# ---- 2. Zenodo record listing: NUS Enth longitudinal (5502441) ----
try:
    rec = json.loads(get('https://zenodo.org/api/records/5502441', 60))
    print('\nZENODO 5502441 (NUS Enth):', rec.get('metadata', {}).get('title'), flush=True)
    for f in rec.get('files', []):
        print('  FILE', f['key'], round(f['size'] / 1e6, 2), 'MB', f['links']['self'], flush=True)
except Exception as e:
    print('Zenodo listing fail', repr(e), flush=True)
