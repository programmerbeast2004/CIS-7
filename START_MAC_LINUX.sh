#!/bin/bash
echo ""
echo " ========================================"
echo "  CIS-7 COSMIC INTELLIGENCE SYSTEM"
echo " ========================================"
echo ""

# Move to script directory
cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo " [ERROR] Python3 not found. Install from https://python.org"
    exit 1
fi

# Move to api folder
cd api

# Install dependencies
echo " [1/3] Installing dependencies..."
pip3 install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo " [ERROR] Failed to install requirements."
    exit 1
fi

# Check required files
echo " [2/3] Checking required files..."

if [ ! -f "../ml/Model/best_svc_Model.pkl" ]; then
    echo ""
    echo " [ERROR] Missing: ml/Model/best_svc_Model.pkl"
    echo " Copy it from your GitHub repo into ml/Model/"
    echo ""
    exit 1
fi

if [ ! -f "../ml/Model/num_pipeline.pkl" ]; then
    echo ""
    echo " [ERROR] Missing: ml/Model/num_pipeline.pkl"
    echo " Copy it from your GitHub repo into ml/Model/"
    echo ""
    exit 1
fi

if [ ! -f "../ml/Dataset/thermoracleTrain.csv" ]; then
    echo ""
    echo " [ERROR] Missing: ml/Dataset/thermoracleTrain.csv"
    echo " Copy it from your GitHub repo into ml/Dataset/"
    echo ""
    exit 1
fi

echo " [3/3] All files found. Starting server..."
echo ""
echo " ----------------------------------------"
echo "  Open your browser at:"
echo "  http://localhost:8000"
echo " ----------------------------------------"
echo ""
echo " Press Ctrl+C to stop the server."
echo ""

# Start server
uvicorn main:app --reload --port 8000
