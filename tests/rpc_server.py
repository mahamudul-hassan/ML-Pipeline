"""Line-delimited JSON RPC wrapper around engine.py, used by tests/ui_test.mjs instead of the Pyodide worker."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import engine

def out(m):
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()

engine.set_progress(lambda m: out({"type": "progress", "data": m}))
out({"type": "status", "status": "ready", "message": "Python engine ready", "versions": json.loads(engine.versions()), "optional": {"xgboost": True, "lightgbm": True}})
for line in sys.stdin:
    m = json.loads(line)
    try:
        r = "{}" if m["fn"] == "__ping" else getattr(engine, m["fn"])(*m["args"])
    except Exception as e:  # noqa: BLE001
        r = json.dumps({"error": f"{type(e).__name__}: {e}"})
    out({"type": "result", "id": m["id"], "result": r})
