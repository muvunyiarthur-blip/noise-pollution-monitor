# Noise Pollution Monitor (K-Means + Streamlit)

A starter project that clusters sound readings into Quiet Zone, Moderate Zone, and Noisy Zone.

## Important data note
`data/noise_data.csv` is **synthetic demonstration data**, not actual measurements. Replace it with real calibrated sound-meter readings before using the outputs for real-world decisions. The categories are relative clusters, not official legal noise thresholds.

## Project structure
- `notebooks/noise_monitor_step_by_step.ipynb` — guided Jupyter notebook
- `app.py` — Streamlit dashboard, prediction form, CSV upload/download
- `train_model.py` — train and save the model from the CSV
- `data/noise_data.csv` — starter sample dataset
- `data/noise_data_with_categories.csv` — created after training
- `artifacts/noise_kmeans_model.joblib` — saved model artifact (generated after training)
- `requirements.txt` — pinned dependencies for consistent deployment

## Run on Windows
Open Command Prompt in this project folder:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If Python 3.12 is not installed, install a Python version supported by the selected package versions. Python 3.14 can have compatibility issues with some scientific packages; use Python 3.12 for this starter setup.

## Use the notebook
Open Jupyter Notebook, open `notebooks/noise_monitor_step_by_step.ipynb`, and run cells from top to bottom. It will create the model artifact.

## CSV format
Required columns:
- `timestamp` — for example `2026-10-22 08:30:00`
- `location` — for example `Main Road`
- `sound_level_db` — numeric calibrated sound pressure level

## GitHub + Streamlit Community Cloud
1. Create a GitHub repository and upload the project files.
2. Ensure `app.py`, `requirements.txt`, `data/`, and `artifacts/noise_kmeans_model.joblib` are committed.
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository, branch (`main`), and main file path `app.py`, then deploy.
5. If the app fails, open the app's Manage/Logs area and check the build error.

Do not upload private or sensitive data. If using GitHub's web upload, include the model artifact and sample CSV.
