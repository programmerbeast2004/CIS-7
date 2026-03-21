"""
CIS-7 Cosmic Intelligence System — FastAPI Backend
===================================================
Drop the ml/ folder next to api/ and run:

    cd CIS/api
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Open http://localhost:8000
"""

import os, re, json, threading, time
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio

# ── sklearn imports (same as notebook) ────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
ML_DIR    = os.path.join(BASE, "..", "ml")
MODEL_DIR = os.path.join(ML_DIR, "Model")
DATA_DIR  = os.path.join(ML_DIR, "Dataset")

# ── Auto-download from HuggingFace if files missing (for cloud deployment) ────
def download_from_gdrive(file_id, dest_path):
    """Download a file from Google Drive using its file ID."""
    import urllib.request, urllib.error

    # Direct download URL for Google Drive
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    print(f"[GDRIVE] Downloading {os.path.basename(dest_path)} ...")
    try:
        # First request — may get a virus scan warning page for large files
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            content = response.read()

        # Check if Google returned a virus scan warning (happens for large files)
        if b"Google Drive - Virus scan warning" in content or b"confirm=" in content[:2000]:
            # Extract confirm token and retry
            import re
            confirm = re.search(rb"confirm=([0-9A-Za-z_]+)", content)
            if confirm:
                token = confirm.group(1).decode()
                url2 = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
                req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req2) as r2:
                    content = r2.read()
            else:
                # Try alternate direct download URL
                url2 = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
                req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req2) as r2:
                    content = r2.read()

        with open(dest_path, "wb") as f:
            f.write(content)

        size_kb = os.path.getsize(dest_path) // 1024
        print(f"[OK] {os.path.basename(dest_path)} downloaded ({size_kb} KB)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {os.path.basename(dest_path)}: {e}")
        return False


def ensure_files():
    """
    Check if ML files exist locally.
    If not (running on cloud), download from Google Drive.

    Set these environment variables on Railway:
      GDRIVE_SVC_MODEL    = Google Drive file ID for best_svc_Model.pkl
      GDRIVE_PIPELINE     = Google Drive file ID for num_pipeline.pkl
      GDRIVE_MODEL_LR     = Google Drive file ID for model_lr.pkl
      GDRIVE_MODEL_DT     = Google Drive file ID for model_dt.pkl
      GDRIVE_MODEL_RF     = Google Drive file ID for model_rf.pkl
      GDRIVE_MODEL_KNN    = Google Drive file ID for model_knn.pkl
      GDRIVE_MODEL_XGB    = Google Drive file ID for model_xgb.pkl
      GDRIVE_MODEL_GB     = Google Drive file ID for model_gb.pkl
      GDRIVE_MODEL_ADA    = Google Drive file ID for model_ada.pkl
      GDRIVE_MODEL_NB     = Google Drive file ID for model_nb.pkl
      GDRIVE_MODEL_ET     = Google Drive file ID for model_et.pkl
      GDRIVE_TRAIN_CSV    = Google Drive file ID for thermoracleTrain.csv
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Map: (local filename, env var name)
    FILE_MAP = [
        (os.path.join(MODEL_DIR, "best_svc_Model.pkl"), "GDRIVE_SVC_MODEL"),
        (os.path.join(MODEL_DIR, "num_pipeline.pkl"),   "GDRIVE_PIPELINE"),
        (os.path.join(MODEL_DIR, "model_lr.pkl"),       "GDRIVE_MODEL_LR"),
        (os.path.join(MODEL_DIR, "model_dt.pkl"),       "GDRIVE_MODEL_DT"),
        (os.path.join(MODEL_DIR, "model_rf.pkl"),       "GDRIVE_MODEL_RF"),
        (os.path.join(MODEL_DIR, "model_knn.pkl"),      "GDRIVE_MODEL_KNN"),
        (os.path.join(MODEL_DIR, "model_xgb.pkl"),      "GDRIVE_MODEL_XGB"),
        (os.path.join(MODEL_DIR, "model_gb.pkl"),       "GDRIVE_MODEL_GB"),
        (os.path.join(MODEL_DIR, "model_ada.pkl"),      "GDRIVE_MODEL_ADA"),
        (os.path.join(MODEL_DIR, "model_nb.pkl"),       "GDRIVE_MODEL_NB"),
        (os.path.join(MODEL_DIR, "model_et.pkl"),       "GDRIVE_MODEL_ET"),
        (os.path.join(DATA_DIR,  "thermoracleTrain.csv"), "GDRIVE_TRAIN_CSV"),
    ]

    all_local = True
    for local_path, env_var in FILE_MAP:
        if os.path.exists(local_path):
            print(f"[LOCAL] {os.path.basename(local_path)} found")
        else:
            all_local = False
            file_id = os.environ.get(env_var, "")
            if file_id:
                download_from_gdrive(file_id, local_path)
            else:
                print(f"[MISSING] {os.path.basename(local_path)} — set env var {env_var}")

    if all_local:
        print("[OK] All files found locally — no downloads needed")

ensure_files()

# ── EXACT label map from notebook ─────────────────────────────────────────────
LABEL_TO_PLANET = {
    0: "Bewohnbar",
    1: "Terraformierbar",
    2: "Rohstoffreich",
    3: "Wissenschaftlich",
    4: "Gasriese",
    5: "Wüstenplanet",
    6: "Eiswelt",
    7: "Toxischeatmosphäre",
    8: "Hohestrahlung",
    9: "Toterahswelt"
}
PLANET_TO_LABEL = {v: k for k, v in LABEL_TO_PLANET.items()}

CLASS_META = {
    "Bewohnbar":          {"en": "Habitable",        "color": "#3DCC7E", "icon": "◉"},
    "Terraformierbar":    {"en": "Terraformable",    "color": "#4A8FD4", "icon": "◈"},
    "Rohstoffreich":      {"en": "Resource-Rich",    "color": "#E8A020", "icon": "◆"},
    "Wissenschaftlich":   {"en": "Scientific",       "color": "#A855F7", "icon": "⬡"},
    "Gasriese":           {"en": "Gas Giant",        "color": "#9B72D4", "icon": "◎"},
    "Wüstenplanet":       {"en": "Desert World",     "color": "#CC8844", "icon": "◐"},
    "Eiswelt":            {"en": "Ice World",        "color": "#88D4F0", "icon": "◇"},
    "Toxischeatmosphäre": {"en": "Toxic Atmosphere", "color": "#E0504A", "icon": "⊗"},
    "Hohestrahlung":      {"en": "High Radiation",   "color": "#FF6B35", "icon": "☢"},
    "Toterahswelt":       {"en": "Dead World",       "color": "#7A8090", "icon": "○"},
}

# ── EXACT column order from notebook ──────────────────────────────────────────
CATEGORICAL_COLS = ['Magnetic Field Strength', 'Radiation Levels']
NUMERIC_BASE     = ['Atmospheric Density', 'Surface Temperature', 'Gravity',
                    'Water Content', 'Mineral Abundance', 'Orbital Period',
                    'Proximity to Star', 'Atmospheric Composition Index']
NUM_COLS         = NUMERIC_BASE + CATEGORICAL_COLS   # exactly as in notebook cell 11

# ── Extract number from Category_X string (exact notebook logic) ───────────────
def extract_number(cat):
    if isinstance(cat, str):
        m = re.search(r'Category_(\d+)', cat)
        return int(m.group(1)) if m else np.nan
    return cat

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="CIS-7 Cosmic Intelligence System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Shared state (updated by background thread) ────────────────────────────────
state = {
    "pipeline":      None,
    "num_cols":      None,
    "models":        {},        # id(int) → {"clf": ..., "metrics": {...}, "name": ...}
    "status":        "BOOTING",
    "log":           [],        # list of {"time": ..., "msg": ..., "level": ...}
    "train_rows":    0,
    "ready_count":   0,
}

def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    entry = {"time": ts, "msg": msg, "level": level}
    state["log"].append(entry)
    print(f"[{ts}] [{level}] {msg}")

# ── BACKGROUND BOOT: loads pipeline + trains all 10 models ────────────────────
def boot_system():
    state["status"] = "LOADING_PIPELINE"

    # ── 1. Load num_pipeline.pkl ──────────────────────────────────────────────
    pkl_path = os.path.join(MODEL_DIR, "num_pipeline.pkl")
    if not os.path.exists(pkl_path):
        log(f"num_pipeline.pkl not found at {pkl_path}", "ERROR")
        state["status"] = "ERROR"; return
    try:
        loaded = joblib.load(pkl_path)
        pipeline, num_cols = loaded if isinstance(loaded, tuple) else (loaded, NUM_COLS)
        state["pipeline"] = pipeline
        state["num_cols"]  = num_cols
        log(f"num_pipeline.pkl loaded · {len(num_cols)} features: {num_cols}")
    except Exception as e:
        log(f"Failed to load pipeline: {e}", "ERROR")
        state["status"] = "ERROR"; return

    # ── 2. Load training data ─────────────────────────────────────────────────
    state["status"] = "LOADING_DATA"
    train_path = os.path.join(DATA_DIR, "thermoracleTrain.csv")
    if not os.path.exists(train_path):
        log(f"thermoracleTrain.csv not found", "ERROR")
        state["status"] = "ERROR"; return

    log("Loading thermoracleTrain.csv …")
    df = pd.read_csv(train_path)
    df = df.dropna(subset=["Prediction"]).copy()
    state["train_rows"] = len(df)
    log(f"Dataset loaded · {len(df)} rows after dropping NaN targets")

    for col in CATEGORICAL_COLS:
        if df[col].dtype == object:
            df[col] = df[col].apply(extract_number)

    df[num_cols] = pipeline.transform(df[num_cols])
    X = df[num_cols].values
    y = df["Prediction"].astype(int).values
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log(f"Train/val split: {len(X_tr)} train / {len(X_val)} val")

    # ── 3. All 10 model definitions (equal treatment — no SVC special casing) ─
    # Saved pkl name for each model (if exists, load it; otherwise train & save)
    model_defs = [
        (0, "Logistic Reg",   "model_lr.pkl",   LogisticRegression(max_iter=500, random_state=42)),
        (1, "Decision Tree",  "model_dt.pkl",   DecisionTreeClassifier(max_depth=10, criterion="entropy", random_state=42)),
        (2, "Random Forest",  "model_rf.pkl",   RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        (3, "KNN",            "model_knn.pkl",  KNeighborsClassifier(n_neighbors=5, weights="distance")),
        (4, "SVC",            "best_svc_Model.pkl", None),   # always load from your pkl
        (5, "XGBoost",        "model_xgb.pkl",  _make_xgb()),
        (6, "Grad. Boost",    "model_gb.pkl",   GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
        (7, "AdaBoost",       "model_ada.pkl",  AdaBoostClassifier(n_estimators=100, random_state=42)),
        (8, "Naive Bayes",    "model_nb.pkl",   GaussianNB()),
        (9, "Extra Trees",    "model_et.pkl",   ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ]

    state["status"] = "LOADING_MODELS"

    for mid, name, pkl_name, clf_template in model_defs:
        pkl_full = os.path.join(MODEL_DIR, pkl_name)
        loaded_clf = None

        # Try loading from saved pkl first
        if os.path.exists(pkl_full):
            try:
                loaded_clf = joblib.load(pkl_full)
                # For SVC the pkl IS the model directly
                # For others it might be a tuple (model, meta) — handle both
                if isinstance(loaded_clf, tuple):
                    loaded_clf = loaded_clf[0]
                m = _metrics(loaded_clf, X_val, y_val)
                state["models"][mid] = {"clf": loaded_clf, "metrics": m, "name": name, "ready": True}
                state["ready_count"] += 1
                log(f"{name} loaded from pkl · acc={m['acc']}%")
                continue
            except Exception as e:
                log(f"{name} pkl load failed ({e}) — will retrain", "WARN")
                loaded_clf = None

        # No pkl or load failed — train from scratch
        if clf_template is None:
            log(f"{name}: pkl not found at {pkl_full} and no template — skipping", "WARN")
            continue

        state["status"] = f"TRAINING_{name.upper().replace(' ', '_')}"
        log(f"Training {name} …")
        t0 = time.time()
        try:
            clf_template.fit(X_tr, y_tr)
            m = _metrics(clf_template, X_val, y_val)
            state["models"][mid] = {"clf": clf_template, "metrics": m, "name": name, "ready": True}
            state["ready_count"] += 1
            log(f"{name} trained · acc={m['acc']}% · {time.time()-t0:.1f}s")
            # Save for next run so it loads instantly
            try:
                joblib.dump(clf_template, pkl_full)
                log(f"{name} saved to {pkl_name}")
            except Exception as e:
                log(f"Could not save {name}: {e}", "WARN")
        except Exception as e:
            log(f"{name} training failed: {e}", "WARN")

    state["status"] = "READY"
    log(f"All {len(state['models'])} models ready · System online", "SUCCESS")

def _make_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=5, random_state=42,
                             use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    except ImportError:
        log("xgboost not installed, using GradientBoosting for slot 5", "WARN")
        return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

def _metrics(clf, X_val, y_val):
    preds = clf.predict(X_val)
    acc   = round(accuracy_score(y_val, preds) * 100, 1)
    prec  = round(precision_score(y_val, preds, average='weighted', zero_division=0) * 100, 1)
    rec   = round(recall_score(y_val, preds, average='weighted', zero_division=0) * 100, 1)
    f1    = round(f1_score(y_val, preds, average='weighted', zero_division=0) * 100, 1)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

# Start boot in background thread
threading.Thread(target=boot_system, daemon=True).start()

# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    # 8 numeric features + 2 categorical (pass raw Category_X strings OR floats)
    atmo_density:   float
    surface_temp:   float
    gravity:        float
    water_content:  float
    mineral_abund:  float
    orbital_period: float
    prox_to_star:   float
    atmo_comp:      float
    mag_field:      str   # e.g. "Category_9" or "9"
    radiation:      str   # e.g. "Category_8" or "8"
    model_id:       Optional[int] = 4

# ── /health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":      state["status"],
        "ready_count": state["ready_count"],
        "models":      {k: {"name": v["name"], "acc": v["metrics"]["acc"]}
                        for k, v in state["models"].items()},
        "train_rows":  state["train_rows"],
    }

# ── /status (SSE stream for real-time log) ─────────────────────────────────────
@app.get("/status-stream")
async def status_stream():
    """Server-Sent Events: streams boot log + status to frontend in real time."""
    async def generate():
        sent = 0
        while True:
            # Send any new log entries
            while sent < len(state["log"]):
                entry = state["log"][sent]
                data = json.dumps({"type": "log", **entry,
                                   "status": state["status"],
                                   "ready": state["ready_count"]})
                yield f"data: {data}\n\n"
                sent += 1
            # Send heartbeat
            hb = json.dumps({"type": "hb", "status": state["status"],
                             "ready": state["ready_count"]})
            yield f"data: {hb}\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

# ── /models ────────────────────────────────────────────────────────────────────
@app.get("/models")
def get_models():
    return {
        k: {"name": v["name"], "metrics": v["metrics"], "ready": v["ready"]}
        for k, v in state["models"].items()
    }

# ── /predict ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictRequest):
    pipeline = state["pipeline"]
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded yet")

    mid = req.model_id
    # -1 = pick best available model by accuracy
    if mid == -1 or mid not in state["models"]:
        ready = [k for k in state["models"] if state["models"][k]["ready"]]
        if not ready:
            raise HTTPException(503, "No models ready yet — still training")
        mid = max(ready, key=lambda k: state["models"][k]["metrics"]["acc"])

    clf = state["models"][mid]["clf"]

    # Build row in exact notebook column order
    row = {
        "Atmospheric Density":         req.atmo_density,
        "Surface Temperature":         req.surface_temp,
        "Gravity":                     req.gravity,
        "Water Content":               req.water_content,
        "Mineral Abundance":           req.mineral_abund,
        "Orbital Period":              req.orbital_period,
        "Proximity to Star":           req.prox_to_star,
        "Atmospheric Composition Index": req.atmo_comp,
        "Magnetic Field Strength":     extract_number(req.mag_field),
        "Radiation Levels":            extract_number(req.radiation),
    }
    df_row = pd.DataFrame([row])[state["num_cols"]]
    X_scaled = pipeline.transform(df_row)

    pred_int = int(clf.predict(X_scaled)[0])
    pred_str = LABEL_TO_PLANET.get(pred_int, str(pred_int))

    # Confidence
    try:
        proba = clf.predict_proba(X_scaled)[0]
        conf  = round(float(proba.max()) * 100, 1)
    except AttributeError:
        try:
            df_val = clf.decision_function(X_scaled)[0]
            score  = float(np.max(np.abs(df_val))) if hasattr(df_val,"__len__") else float(abs(df_val))
            conf   = round(min(99.0, 55 + score * 10), 1)
        except Exception:
            conf = state["models"][mid]["metrics"]["acc"]

    meta = CLASS_META.get(pred_str, {"en": pred_str, "color": "#C8922A", "icon": "●"})
    return {
        "predicted_class": pred_str,
        "predicted_int":   pred_int,
        "english_label":   meta["en"],
        "color":           meta["color"],
        "icon":            meta["icon"],
        "confidence":      conf,
        "model_id":        mid,
        "model_name":      state["models"][mid]["name"],
        "model_acc":       state["models"][mid]["metrics"]["acc"],
    }

# ── Serve index.html ───────────────────────────────────────────────────────────
@app.get("/")
def ui():
    f = os.path.join(BASE, "index.html")
    return FileResponse(f) if os.path.exists(f) else {"msg": "Place index.html next to main.py"}