# CIS-7 · Cosmic Intelligence System
### Galactic Classification Challenge · Planet Survival AI

---

## STEP 1 — Place your files

Copy these 3 files from your GitHub repo into this folder structure:

```
CIS_FINAL/
│
├── ml/
│   ├── Model/
│   │   ├── best_svc_Model.pkl        ← COPY HERE from GitHub
│   │   └── num_pipeline.pkl          ← COPY HERE from GitHub
│   │
│   ├── Dataset/
│   │   └── thermoracleTrain.csv      ← COPY HERE from GitHub
│   │
│   ├── Code/
│   │   └── app.ipynb                 (optional)
│   └── Predictions/
│       └── Final_predictions.csv     (optional)
│
└── api/
    ├── main.py       ← backend (do not move)
    ├── index.html    ← frontend (do not move)
    └── requirements.txt
```

Your GitHub repo path:
- `Cosmic-Classifier/ml/Model/best_svc_Model.pkl`
- `Cosmic-Classifier/ml/Model/num_pipeline.pkl`
- `Cosmic-Classifier/ml/Dataset/thermoracleTrain.csv`

---

## STEP 2 — Run

### Windows
Double-click `START_WINDOWS.bat`

### Mac / Linux
```bash
chmod +x START_MAC_LINUX.sh
./START_MAC_LINUX.sh
```

### Manual (any OS)
```bash
cd CIS_FINAL/api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

---

## STEP 3 — Open browser

Go to: **http://localhost:8000**

⚠️ Do NOT open index.html directly as a file — it must go through the server.

---

## What happens when you run it

| Step | What the system does |
|------|---------------------|
| Boot screen | Connects to backend, streams live log |
| 0-5 sec | Loads your num_pipeline.pkl + best_svc_Model.pkl |
| 5-10 sec | Reads thermoracleTrain.csv (60,000 rows) |
| 10-60 sec | Trains 9 other models in background (LR, DT, RF, KNN, XGB, GB, ADA, NB, ET) |
| Live | Model tabs unlock one-by-one as each finishes, showing real accuracy |
| Ready | All 10 models available, galaxy scene running, planet morphs on classification |

---

## Troubleshooting

**"Module not found: xgboost"**
```bash
pip install xgboost
```

**"Pipeline not loaded" error**
Make sure `num_pipeline.pkl` is in `ml/Model/` not anywhere else.

**Models tab shows "…" forever**
Backend is not running. Check the terminal — look for errors.

**Page is blank / galaxy not showing**
Open DevTools (F12) → Console. If you see a Three.js error, try a different browser (Chrome works best).

**Backend offline warning in UI**
The UI still works in simulation mode — just not using your real model. Start the backend with `uvicorn main:app --reload --port 8000`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML + CSS + Three.js (r128) |
| Backend | FastAPI + Uvicorn |
| ML Pipeline | scikit-learn, XGBoost |
| Models | SVC, RF, XGB, GB, KNN, DT, LR, ADA, NB, ET |
| Dataset | thermoracleTrain.csv — 60,000 rows, 10 features |
| Best Model | SVC — 92% accuracy (GridSearchCV tuned) |

---

Built for the Galactic Classification Challenge (GCC) · 2547
