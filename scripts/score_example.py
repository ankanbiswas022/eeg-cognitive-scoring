"""Call the running service with one subject's task recording and print the score.

    uvicorn eegscore.serve.app:app --port 8000 &
    python scripts/score_example.py data/raw/eegmat/Subject00_2.edf
"""
import json
import sys

import mne
import requests

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/eegmat/Subject00_2.edf"
raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
raw.pick([c for c in raw.ch_names if c.startswith("EEG ") and "A2-A1" not in c])
payload = {"sfreq": raw.info["sfreq"], "ch_names": raw.ch_names,
           "data": (raw.get_data() * 1e6).tolist(), "units": "uV"}
r = requests.post("http://localhost:8000/score", json=payload, timeout=60)
print(json.dumps(r.json(), indent=2)[:2500])
