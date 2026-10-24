# HelioWeave — 3D Research Edition
## Real Time Solar Forecasting and Intelligent EV Scheduling Using LSTM

A redesigned **Flask + Python + HTML/CSS/JavaScript** application with a project-specific **Three.js 3D solar campus**. Includes actual trained **LightGBM, XGBoost, CatBoost, KNN and Stacking** models, your exact 20-feature Kaggle dataset, and the original two-LSTM forecasting/scheduling workflow.

## Start in VS Code — Git Bash on Windows
1. Extract the ZIP. Open the inner `HelioWeave_3D` folder in VS Code (the folder containing `app.py`).
2. Choose Terminal → New Terminal → Git Bash.
3. Run:

```bash
py -3.14 --version
py -3.14 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tested.txt
python app.py
```

Open **http://127.0.0.1:5000**. Keep the terminal running. Press **Ctrl+C** to stop.

**Models and datasets are already included. No training, API key, npm, GPU or Node.js installation is needed to launch.** Python packages must be installed once, which needs internet. Browser assets are bundled locally, so the running app does not depend on a CDN.

The project was trained and tested with **64-bit CPython 3.14.7 on Linux**. The commands above target Windows Git Bash. Windows itself was not the test operating system. Compatible Windows x64 wheels for all nine direct pinned dependencies were confirmed through PyPI; see `reports/windows_wheel_check.json`. Use standard CPython, not a free-threaded build. `py -3.14` selects your installed 3.14 version; the first command should show 3.14.7 if that is the exact version you want.

In VS Code select Ctrl+Shift+P → Python: Select Interpreter → `.venv/Scripts/python.exe`.

### Later launches
```bash
source .venv/Scripts/activate
python app.py
```

### If activation is inconvenient
```bash
./.venv/Scripts/python.exe -m pip install -r requirements-tested.txt
./.venv/Scripts/python.exe app.py
```

### Linux, macOS or WSL Bash
```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tested.txt
python app.py
```

Do not mix WSL virtual environments with Windows Git Bash environments.

## What you can do
1. **Campus overview:** orbit and zoom around a 3D model of solar arrays, inverters, EVs, charging bays and grid equipment. Camera presets focus on the campus, arrays and EV bays.
2. **Prediction studio:** enter all 20 parameters, load real held-out dataset observations, and execute all five trained models. Upload a CSV for batch inference and download the predictions.
3. **EV dispatch:** enter 1–12 EVs with battery capacity, initial/target SOC, arrival/departure and charger power; set PV scale, grid limit and building demand. The planner minimizes unmet demand first, then electricity cost. Export the schedule and move the hourly slider to see matching energy-flow activity in the 3D scene.
4. **Model observatory:** compare actual held-out RMSE/R², inspect validation permutation importance and understand the stacking method.
5. **Dataset explorer:** inspect the exact CSV and all feature names; download the original data.
6. **LSTM research:** run the original timestamped solar forecast plus learned EV charging controller, with immediate, rolling-optimization and oracle comparisons.

The campus is a **3D illustration**, not a surveyed digital twin or a hardware-connected installation. It uses restrained materials, panel cell geometry, vehicles and power-flow animation. Animation can be paused, honors reduced-motion preferences, and degrades gracefully if WebGL is unavailable. Four display vehicles illustrate charging infrastructure; the planner supports up to twelve actual scenario requests.

## Dataset location
The exact new schema is already in **`data/spg.csv`**. Do not paste it into a Python file.

Source: https://www.kaggle.com/datasets/stucom/solar-energy-power-generation-dataset

It contains 4,213 rows, your 20 input columns, and `generated_power_kw` as the target. See `data/DATASET_CARD.md` for attribution, rights, units and limitations. The original LSTM dataset is separately included under `lstm_research/data/raw/`.

## Retraining
Stop the server first, activate your environment, then run:
```bash
python train.py
python -m unittest discover -s tests -v
python app.py
```

To use a replacement CSV with the same exact schema:
```bash
python train.py --csv "C:/Users/YourName/Downloads/your_solar_data.csv"
```

This regenerates the models, metrics and presets. The dataset browser still shows `data/spg.csv`; to make a new dataset authoritative throughout the app, replace that file and run `python train.py` without `--csv`. Restart the server to clear its model cache. Training and tests use CPU.

To retrain the original LSTM module:
```bash
cd lstm_research
python train.py --epochs 25
python evaluate.py --sensitivity
cd ..
python app.py
```

## Important research distinction
`spg.csv` has **no timestamps**. These five models estimate power from weather and solar geometry; they do not establish a verified future forecasting horizon. Evaluation uses contiguous source-row blocks (60% train / 20% validation / 20% test), not a claimed chronological split. The Stacking Model uses four expanding out-of-fold training blocks, with preprocessing fitted within each fold, and a Ridge meta-model.

The separate LSTM module uses timestamped measured plant data and chronological evaluation. The EV dispatch module uses an **illustrative hourly weather profile**, user-supplied requests, assumed 92% charging efficiency and illustrative tariffs. It reports estimated planned cost and unmet demand. It is not live operation. There is no patent or universal-performance claim.

## Project map
| Path | Purpose |
|---|---|
| `app.py` | Flask server and APIs |
| `train.py` | Five-model training, OOF stacking and evaluation |
| `core/schema.py` | Exact 20-column schema and validation |
| `core/modeling.py` | Real algorithm definitions and inference |
| `core/ev.py` | Service-first optimization and scenario generation |
| `templates/index.html` | Six-screen application |
| `static/style.css` | Responsive visual design |
| `static/app.js` | Forms, API calls, charts, exports and navigation |
| `static/scene.js` | Procedural Three.js campus and animation |
| `static/vendor/` | Bundled Three.js and its MIT license |
| `data/spg.csv` | Original requested-schema Kaggle dataset |
| `models/ensemble.joblib` | Trained four base models and Ridge stack |
| `reports/` | Actual metrics, predictions and test evidence |
| `docs/PROJECT_GUIDE.md` | Simple explanation, viva answers and demonstration flow |
| `docs/RESULTS.md` | Measured results and limitations |
| `lstm_research/` | Original two-LSTM research source, data and artifacts |

Load only your own trusted model artifacts. The application does not accept uploaded model files.

## Troubleshooting
- **`py` is not found:** install 64-bit Python and reopen VS Code. If `python --version` already shows your intended version, use `python -m venv .venv`.
- **Missing modules:** run `python -m pip install -r requirements-tested.txt` using the same activated interpreter.
- **Package installation errors:** update pip first; use standard 64-bit CPython. Do not combine old and new virtual environments. Check the first actual error in the terminal.
- **Windows DLL errors:** install the Microsoft Visual C++ Redistributable matching your x64 Python installation, then reopen the terminal.
- **macOS LightGBM OpenMP error:** the official LightGBM package may require a system OpenMP runtime; install `libomp` with your package manager if its error requests it.
- **Port already in use:** run `python app.py --port 5001`, then open http://127.0.0.1:5001.
- **Blank 3D scene:** use Chrome or Edge with hardware acceleration/WebGL enabled. The forms still work if WebGL is unavailable. Do not open `index.html` by double-clicking; use the Flask URL.
- **Slow 3D rendering:** pause motion; zoom less; close GPU-heavy browser tabs. The scene caps pixel density and does not need a discrete GPU.
- **No model files:** run `python train.py`.
- **Latest Python versions:** recreate the environment and try `requirements.txt`; rerun training/tests if package versions change. Future unreleased versions are not guaranteed. `requirements-tested.txt` is the reproducible setup used for the bundled artifacts.

The server listens on localhost with debug mode disabled. It is a research/development application, not an authenticated production service.
