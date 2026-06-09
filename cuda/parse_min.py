"""Parse run_gist.sh output -> one line: min_ms passed max_abs.  Usage: ./run_gist.sh <rev> big 2>&1 | python3 parse_min.py"""
import sys, json, re
t = sys.stdin.read()
m = re.search(r'"stdout": "(.*?)",\n\s*"stderr"', t, re.S)
inner = None
if m:
    raw = m.group(1).encode().decode('unicode_escape')
    raw = raw[raw.find('{'):]
    try: inner = json.loads(raw)
    except Exception: inner = None
if inner:
    g = inner["impls"]["gen-cuda"]
    print(f'min_ms={g["performance"]["latency_ms"]["min"]:.4f} passed={g["correctness"]["passed"]} max_abs={g["correctness"]["max_abs_error"]}')
else:
    mins = re.findall(r'"min": ([0-9.]+)', t)
    pas = re.findall(r'"passed": (true|false)', t)
    mae = re.findall(r'"max_abs_error": ([0-9.eE+-]+)', t)
    print(f'min_ms={mins[0] if mins else "?"} passed={pas[0] if pas else "?"} max_abs={mae[0] if mae else "?"}')
