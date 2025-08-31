Email Compliance Classifier — Quick UI

This folder contains a small Streamlit demo UI to interact with the trained classifier.

Files:
- `ui_app.py` — Streamlit app. Launch with `streamlit run ui_app.py`.
- `requirements.txt` — Adds `streamlit`.

Run locally (Windows, bash):

```bash
# (optional) create and activate environment
# conda create -n hr-classifier python=3.10 -y
# conda activate hr-classifier

pip install -r requirements.txt
# If you don't have transformers/torch installed, install them too:
# pip install transformers torch

streamlit run ui_app.py
```

Notes:
- The app loads the model from `models/final_model` by default. If your model is elsewhere, provide the path in the UI.
- Loading the model can take a few seconds on CPU.
