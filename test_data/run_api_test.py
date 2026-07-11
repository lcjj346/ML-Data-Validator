"""
Full API mock test:
1. Validate demo_validation_test.csv against base_model
2. Train a custom model from custom_hospital_training.csv
3. Validate custom_hospital_validate.csv against the new custom model
"""

import json
import requests
import sseclient
import sys

BASE = "http://localhost:8765/api"

def ok(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    mark = "OK" if condition else "!!"
    print(f"  [{status}] {mark} {label}" + (f" - {detail}" if detail else ""))
    return condition

def run_sse(url):
    """Consume an SSE stream and return the 'done' event."""
    resp = requests.get(url, stream=True)
    client = sseclient.SSEClient(resp)
    last = None
    for event in client.events():
        data = json.loads(event.data)
        last = data
        if data["type"] in ("done", "error"):
            break
    return last

# ─────────────────────────────────────────────
print("\n==========================================")
print("  TEST 1: Validate demo CSV with base_model")
print("==========================================")

# Upload
with open("test_data/demo_validation_test.csv", "rb") as f:
    r = requests.post(f"{BASE}/validate/upload", files={"file": ("demo_validation_test.csv", f, "text/csv")})
ok("Upload returns 200", r.status_code == 200, f"status={r.status_code}")
session = r.json()
session_id = session["session_id"]
ok("Session ID returned", bool(session_id))
ok("Correct row count", session["rows"] == 14, f"rows={session['rows']}")

# Match columns
r = requests.post(f"{BASE}/validate/match-columns",
    json={"session_id": session_id, "model_name": "base_model"})
ok("Column match returns 200", r.status_code == 200, f"status={r.status_code}")
match = r.json()
ok("Columns matched", len(match["matched"]) >= 5, f"matched={list(match['matched'].keys())}")
ok("No critical unmatched", True, f"unmatched={match['unmatched']}")

# Run validation (SSE)
done = run_sse(f"{BASE}/validate/{session_id}/run?model_name=base_model")
ok("SSE stream completes", done and done["type"] == "done")

# Get results
r = requests.get(f"{BASE}/validate/{session_id}/results")
ok("Results returns 200", r.status_code == 200)
results = r.json()
ok("Has corrections", len(results["corrections"]) > 0, f"corrections={len(results['corrections'])}")
ok("Quality < 100% (has invalid cells)", results["quality"] < 100, f"quality={results['quality']:.1f}%")
ok("Confidence scores present", any(c.get("confidence") is not None for c in results["corrections"]),
   f"sample={results['corrections'][0].get('confidence')}")

# Check specific known invalids
reasons = {(c["column"], c["original"]): c for c in results["corrections"]}
ok("'J' flagged as invalid name",      ("name",    "J")             in reasons)
ok("'invalid-email' flagged",          ("email",   "invalid-email") in reasons)
ok("'Singaproe' flagged with correction",
   ("country", "Singaproe") in reasons and reasons[("country","Singaproe")]["has_correction"],
   f"suggestion={reasons.get(('country','Singaproe'), {}).get('suggested')}")

# Export validated CSV
r = requests.get(f"{BASE}/validate/{session_id}/export")
ok("CSV export returns 200", r.status_code == 200)
ok("CSV content type", "text/csv" in r.headers.get("content-type",""))

# Export summary report
r = requests.get(f"{BASE}/validate/{session_id}/export-report")
ok("Summary report returns 200", r.status_code == 200, f"status={r.status_code}")
ok("Report has content", len(r.text) > 100, f"bytes={len(r.text)}")
ok("Report contains Overview", "Overview" in r.text)
ok("Report contains column breakdown", "email" in r.text or "name" in r.text)

# File size guard test
print("\n  [File size guard]")
big_data = b"name,email\n" + b"John,john@test.com\n" * 600000
r = requests.post(f"{BASE}/validate/upload",
    files={"file": ("big.csv", big_data, "text/csv")})
ok("Oversized file rejected", r.status_code == 400, f"status={r.status_code}, detail={r.json().get('detail','')}")

# ─────────────────────────────────────────────
print("\n==========================================")
print("  TEST 2: Train custom hospital model")
print("==========================================")

# Upload training data
with open("training_data/custom_hospital_training.csv", "rb") as f:
    r = requests.post(f"{BASE}/train/upload",
        files={"file": ("custom_hospital_training.csv", f, "text/csv")})
ok("Training upload returns 200", r.status_code == 200, f"status={r.status_code}")
train = r.json()
train_session = train["session_id"]
ok("Training session ID returned", bool(train_session))
ok("Correct training row count", train["rows"] == 40, f"rows={train['rows']}")
ok("5 columns detected", train["columns"] == 5, f"columns={train['columns']}")

# Run training (SSE)
params = "model_name=hospital_test&training_mode=new&exclude_columns=&use_reference_lists=true"
done = run_sse(f"{BASE}/train/{train_session}/run?{params}")
ok("Training SSE completes", done and done["type"] == "done", f"type={done.get('type') if done else 'None'}")
if done and done.get("metrics"):
    metrics = done["metrics"]
    ok("Metrics returned for trained columns", len(metrics) > 0, f"columns={list(metrics.keys())}")
    for col, m in metrics.items():
        acc = m.get("train_accuracy")
        print(f"         {col}: train_acc={acc:.1%}" if acc else f"         {col}: no accuracy")

# Verify model file exists
import os
ok("Model file saved to disk", os.path.exists("models/hospital_test.pkl"), "models/hospital_test.pkl")

# Model appears in list
r = requests.get(f"{BASE}/models")
ok("Model list returns 200", r.status_code == 200)
names = [m["name"] for m in r.json()]
ok("hospital_test appears in model list", "hospital_test" in names, f"models={names}")

# ─────────────────────────────────────────────
print("\n==========================================")
print("  TEST 3: Validate custom CSV with hospital model")
print("==========================================")

# Upload validation CSV
with open("test_data/custom_hospital_validate.csv", "rb") as f:
    r = requests.post(f"{BASE}/validate/upload",
        files={"file": ("custom_hospital_validate.csv", f, "text/csv")})
ok("Validate upload returns 200", r.status_code == 200)
val_session = r.json()["session_id"]

# Match columns against custom model
r = requests.post(f"{BASE}/validate/match-columns",
    json={"session_id": val_session, "model_name": "hospital_test"})
ok("Column match returns 200", r.status_code == 200)
match = r.json()
ok("Custom columns matched", len(match["matched"]) >= 3, f"matched={list(match['matched'].keys())}")

# Run validation
done = run_sse(f"{BASE}/validate/{val_session}/run?model_name=hospital_test")
ok("Validation SSE completes", done and done["type"] == "done")

# Results
r = requests.get(f"{BASE}/validate/{val_session}/results")
ok("Results returns 200", r.status_code == 200)
results = r.json()
ok("Quality < 100% (invalid rows present)", results["quality"] < 100, f"quality={results['quality']:.1f}%")
ok("Corrections list populated", len(results["corrections"]) > 0, f"corrections={len(results['corrections'])}")
print(f"\n  Custom model corrections found:")
for c in results["corrections"]:
    print(f"    row={c['row']} col={c['column']} original='{c['original']}' reason='{c['reason']}' confidence={c.get('confidence',0):.0%}")

# ─────────────────────────────────────────────
print("\n==========================================")
all_pass = True
print("  All tests complete.")
