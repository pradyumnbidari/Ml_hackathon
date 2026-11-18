# PathSnap — Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR Campus).

PathSnap takes a noisy GPS trace and a road network and predicts which road
segment (arc) the vehicle was on at every GPS point. It also returns a
connected route.

**Pipeline:** nearby candidate roads (≤10 within 60 m) → 23 features → LightGBM
(Optuna-tuned) scores each candidate → HMM + Viterbi picks a connected sequence
→ stitched route.

**Student-test results:** Top-1 accuracy 91.2% for the classifier alone, 93.8%
with the HMM. Route length F1 is 98.3%.

---

## 1. Setup (once)

Requires Python 3.10+.

```bash
cd pathsnap
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

If a pinned version fails to install, delete its `==version` in
`requirements.txt` and try again.

**Data:** copy the `train` and `student_test` folders from the given *Student
folder* into `data/raw/`:

```
pathsnap/data/raw/train/00000001/00000001.track ...
pathsnap/data/raw/student_test/00000000/00000000.track ...
```

The trained model is already in `artifacts/`. **You do not need to retrain to
run the demo.**

---

## 2. The demo (two laptops)

Both laptops must be on the same Wi-Fi or hotspot.

### Laptop 1: API + model

1. Put the 10 evaluator trips into `data/eval/`, one folder per trip, the same
   layout as `student_test`.
2. Start the API:

   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

3. Find this laptop's IP address (`ipconfig` on Windows, `ifconfig` on
   Mac/Linux), for example `192.168.1.5`.
4. If Windows asks, allow Python through the firewall.
5. **Leave this terminal open and visible.** It prints a log line for every
   request (trip, points, accuracy). The evaluator will ask to see these logs.

### Laptop 2: UI

Copy the `ui` folder to Laptop 2. Python is the only thing it needs.

```bash
cd ui
python -m http.server 8080
```

Open a browser at:

```
http://localhost:8080/?api=http://192.168.1.5:8000
```

Use Laptop 1's IP. The 10 trips appear on the left. Click one: the map shows
the GPS points (blue) and the predicted route (red), and the right panel shows
the accuracy computed on Laptop 1.

### Practise before the day

Run everything on one laptop, using the student-test trips as stand-ins:

```bash
# terminal 1  (Windows PowerShell: $env:PATHSNAP_EVAL_DIR="data/raw/student_test")
PATHSNAP_EVAL_DIR=data/raw/student_test uvicorn api.main:app --port 8000
# terminal 2
cd ui && python -m http.server 8080
# browser
http://localhost:8080/?api=http://localhost:8000
```

### If something goes wrong

| Problem | Fix |
|---|---|
| UI says "Could not reach" | Wrong IP, firewall blocking port 8000, or the laptops are on different networks. On Laptop 2, test `http://<IP>:8000/health` in a browser. |
| No trips listed | `data/eval/` is empty or has the wrong layout (each trip needs its own folder). |
| Map has no background | Laptop 2 has no internet, so map tiles can't load. The route still draws. |

---

## 3. Re-running the experiments (optional)

Run these in order from the `pathsnap` folder. Times are for a 2-core laptop.

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `python scripts/01_eda.py` | EDA: data-quality checks, plots, Folium maps → `results/eda/` | 1.5 min |
| 2 | `python scripts/02_build_features.py` | Labels, candidates, 23 features → `data/processed/` | 30 s |
| 3 | `python scripts/03_train_models.py` | 8 models, 5-fold GroupKFold + student-test → `results/models/` | 35 min |
| 4 | `python scripts/04_learning_curves.py` | Train/validation loss curves (MLP, LightGBM) | 6 min |
| 5 | `python scripts/05_bayes_opt.py` | Optuna Bayesian tuning, 60 trials → `results/optuna/` | 30 min |
| 6 | `python scripts/05b_feature_ablation.py` then `05c_plot_feature_selection.py` | Feature selection by ablation | 2 min |
| 7 | `python scripts/06_hmm_and_final_eval.py` | HMM tuning, classifier vs HMM, maps, exports `artifacts/` | 10 min |
| 8 | `python scripts/07_summarise.py` | Comparison table and report figures | 3 min |
| 9 | `python report/build_report.py` | Builds the PDF report (needs `pip install playwright` + Chrome) | 1 min |

`05_bayes_opt.py` saves trials to a database and resumes if stopped. Set
`N_TRIALS=10` for a quick run.

---

## Folder layout

| Folder | Contents |
|---|---|
| `src/pathsnap/` | Library: readers, geometry, road network, labels, features, models, metrics, HMM, pipeline, maps |
| `scripts/` | Numbered experiment scripts (table above) |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/index.html` | Evaluator UI (Laptop 2); Leaflet is bundled in `ui/vendor/` |
| `artifacts/` | `final_model.joblib`, `pathsnap_config.json`, all 8 trained models |
| `results/` | All tables, figures, Folium maps, Optuna trials, API demo logs |
| `report/` | Technical report PDF and the script that builds it |
| `data/eval/` | Empty; evaluator trips go here on Laptop 1 |

**No leakage:** the API predicts from `.track/.nodes/.arcs` only, then reads
`.route` on the server to score. The reference route is never sent to the UI.
No training or tuning script reads `data/eval/`.

**AI-assistance disclosure:** this project was developed with help from an AI
assistant (Claude). All numbers come from running these scripts.
