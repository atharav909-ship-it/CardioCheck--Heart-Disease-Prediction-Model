# CardioCheck — redesigned 10-feature educational demo

## IMPORTANT: install your NEW trained model
The original ZIP contained an older 25-feature model. It is deliberately NOT included because it does not match the new 10-feature assessment. Copy your newly trained `best_heart_disease_model.joblib` into this folder (next to `app.py`). **Without it, the interface loads but predictions return HTTP 503.**

The model must be the exact 10-feature fitted Pipeline from your notebook. Expected feature order:

`max_heart_rate_achieved`, `exercise_induced_angina`, `st_depression`, `age`, `ldl`, `hdl`, `resting_bp_systolic`, `exercise_minutes_per_week`, `hba1c`, `bmi`.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Visit http://127.0.0.1:8000/ . Check http://127.0.0.1:8000/api/status to confirm model readiness.

## Deploy on Render

- Commit all files **including your new trusted `.joblib` artifact** to a private repository (or provide it securely through your deployment pipeline).
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Use a compatible Python/scikit-learn version with the environment used to train the artifact. The pinned versions here are inherited from the original project; update them if the notebook used different versions.
- Verify `/api/status` returns `model_ready: true`, then test `/api/predict` with realistic measured values.

## Notes

- The UI uses **all 10 exact features** from the reduced Logistic Regression experiment; it does not invent values for removed inputs.
- Numeric `exercise_induced_angina` is encoded 0/1, consistent with the reduced-model experiment assumption. Verify this matches your CSV.
- Input bounds are basic plausibility checks, **not** medical reference intervals.
- The displayed model probability is **not a clinically validated risk percentage**.
- The earlier reported accuracy 88.78% and ROC-AUC 0.9479 are notebook test results; dataset provenance, leakage risk and external validation remain unresolved.
- No patient history is saved by this demo. Avoid entering real identifiable patient data.
- Never load untrusted joblib/pickle files.


## Enhanced dashboard
Four panels restored: Dashboard, Vitals, Predictions, Settings. Charts show entered measurements, not medical risk contributions. Gauge shows uncalibrated model class-1 probability. Session history is memory-only. Chart.js requires internet access to its CDN. The new 10-feature joblib model must be provided separately; the old 25-feature model is incompatible.


## Important troubleshooting

- This ZIP does **not** contain your newly trained model. Place the 10-feature `best_heart_disease_model.joblib` next to `app.py`. The older 25-feature model will be rejected.
- Start with `uvicorn app:app --reload` and open http://127.0.0.1:8000 (not the HTML file directly).
- Check the model status at the top. If it says unavailable, prediction will show an inline error and preserve all entered values.
- Dashboard, Vitals, Predictions and Settings navigation works without a model; Vitals and Predictions populate after a successful assessment.
- Refresh with Ctrl+Shift+R after replacing the project to clear cached CSS/JS.
