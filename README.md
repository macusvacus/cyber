# Network IDS (CyberShield)

Small network intrusion detection demo with a Streamlit dashboard.

Contents
- `simulator.py` — synthetic network traffic generator (writes `data/network_logs.csv`).
- `detector.py` — rule-based detection engine.
- `classifier.py` — ML classifier training and prediction (saves `models/` artifacts).
- `responder.py` — responder that logs alerts and blocks IPs (writes `logs/`).
- `dashboard.py` — Streamlit dashboard to view live events.

Quickstart (local)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Generate the dataset and run the pipeline:

```powershell
python simulator.py
python detector.py
python classifier.py
python responder.py
```

3. Run the dashboard (Streamlit):

```powershell
streamlit run dashboard.py
```

Deployment (Streamlit Cloud)

1. Push this repository to GitHub (see instructions below).
2. On https://share.streamlit.io create a new app and connect your GitHub repo.
3. Set the run command to:

```
streamlit run dashboard.py
```

Notes
- Do not commit `data/`, `logs/`, `models/`, or your virtual environment. A `.gitignore` is included.
- `dashboard.py` must be launched with Streamlit (`streamlit run`) — running with `python` shows bare-mode warnings.

License
MIT
