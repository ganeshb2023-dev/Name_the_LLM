# Name That LLM — ML inference service

This service runs a scikit-learn character/word TF-IDF + LinearSVC classifier trained on the provided `train.csv`.

## Contents
- `data/train.csv`: labelled training data supplied by the project owner.
- `data/test.csv`: test file supplied by the project owner (the uploaded file has only 3 rows).
- `model/classifier.joblib`: fitted model artifact.
- `model/validation_metrics.json`: grouped holdout evaluation.
- `train.py`: reproducible grouped validation + full-data training script.
- `app.py`: FastAPI inference API.

## Run locally
Use Python 3.11 and install dependencies:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Check `http://127.0.0.1:8000/health`. Test:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"A sufficiently long passage that the classifier can process.\"}"
```
Optional API-key protection: set `CLASSIFIER_API_KEY` in the service environment. If set, requests must include `Authorization: Bearer <key>`.

## Retrain
Replace `data/train.csv` with the official labelled file, then run:
```bash
python train.py
```
The script uses `GroupShuffleSplit` on `prompt_id` for validation and then trains the final model on all labelled rows. Do not use the competition test labels for training or tuning.

## Deploy the API
Deploy this folder as a Docker web service (for example, on Render). Use `/health` as the health-check endpoint. Set `CLASSIFIER_API_KEY` to a long random secret in the backend environment if you want to require authentication. The frontend server-side proxy sends this key from its `CLASSIFIER_API_KEY` environment variable.

## Score interpretation
The API returns normalized softmax-transformed LinearSVC decision scores for display. These are **relative scores, not calibrated probabilities**, and should not be interpreted as proof of authorship. The model is a baseline, not a guaranteed high-accuracy detector.
