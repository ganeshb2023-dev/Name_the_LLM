# Name That LLM — Origin AI

A Next.js frontend plus a real Python inference API for the Kaggle **Name That LLM** task.

## Project layout
- `app/`, `components/`, `lib/`: Next.js web UI and API proxy.
- `ml-service/app.py`: FastAPI endpoint serving actual classifier predictions.
- `ml-service/model/classifier.joblib`: trained word/character TF-IDF + LinearSVC model.
- `ml-service/model/validation_metrics.json`: grouped validation metrics.
- `ml-service/train.py`: reproducible training and grouped holdout evaluation.
- `ml-service/data/train.csv`: labelled data uploaded for this project.
- `ml-service/data/test.csv`: uploaded test file (3 rows).

## Important model limitations
The included baseline was trained on 2,699 labelled examples across 18 classes. A single prompt-grouped 20% holdout achieved **29.1% balanced accuracy** (chance level for 18 balanced classes is about 5.6%). This is a useful working baseline, not a guarantee of high leaderboard performance. Some labels are very similar and this dataset is small. Improve it with repeated grouped validation and experiments before making strong claims.

The uploaded `test.csv` contains only **3 rows**, so it is not enough to create a full Kaggle competition submission if the official test set is larger. The web app can still classify arbitrary user passages using the trained model.

The API's displayed scores are normalized transformations of LinearSVC decision scores, **not calibrated probabilities**. They are relative model scores, not proof of authorship.

## Run the web app locally
Requirements: Node.js 20.9+ (Node 22 recommended), npm.
```bash
npm install
cp .env.example .env.local
npm run dev
```
Open http://localhost:3000. Until `CLASSIFIER_API_URL` is configured, the UI will show an honest model-connection message.

## Run the classifier API locally
Use Python 3.11. From `ml-service`:
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Health: http://127.0.0.1:8000/health

If you set `CLASSIFIER_API_KEY`, the API requires `Authorization: Bearer <key>`. Leave it unset for local testing only.

## Deploy the ML API
Deploy `ml-service` as a Docker web service on a host such as Render:
1. Create a new web service from this repository.
2. Set the root directory to `ml-service`.
3. Choose Docker runtime.
4. Set a long random secret as `CLASSIFIER_API_KEY` in the backend environment.
5. Deploy and wait for `/health` to return `{"status":"ok",...}`.
6. Copy the public HTTPS service URL.

A `render.yaml` blueprint is included as a starting point. Review the plan/cost options on your host before deploying.

## Deploy the frontend to Vercel
1. Push the project folder to GitHub.
2. Import the repository into Vercel and set the project root to the `name-that-llm` folder (the folder containing `package.json`).
3. Add Vercel environment variable `CLASSIFIER_API_URL` with the ML API base URL, for example `https://your-api-host.example.com` (no `/predict` suffix).
4. Add `CLASSIFIER_API_KEY` with the same secret configured in the backend.
5. Redeploy and test with a passage at least 20 characters long.

The frontend server-side route calls `${CLASSIFIER_API_URL}/predict`; the secret is not exposed to browser JavaScript. Do not name it `NEXT_PUBLIC_*`.

## Retrain the classifier
Replace `ml-service/data/train.csv` with the official labelled training file, then run `python train.py` from `ml-service`. The script evaluates a prompt-grouped holdout using `prompt_id`, saves metrics, and retrains on all labelled rows. Do not use test labels for training or model selection.

## API contract
`POST /predict`
```json
{"text":"A sufficiently long passage to classify."}
```
Response contains `label`, `scores` (top five), and `score_type`. The web proxy accepts score values in `[0, 1]` and displays them as relative percentages.
