# Telco Customer Churn Project

This repository has two parts:

1. `churn-basic/` — MongoDB + Python + scikit-learn pipeline
2. `dashboard/` + `dashboard-api/` — React + Vite + Tailwind UI over the same MongoDB data

## 1. Train the models (required first)

MongoDB must be running locally (`mongodb://localhost:27017/`).

```powershell
cd churn-basic
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python load_data.py
python train.py
python query_results.py
```

Full ML notes and the measured metrics are in `churn-basic/README.md`.

## 2. Start the dashboard

Terminal A:

```powershell
cd dashboard-api
npm install
npm start
```

Terminal B:

```powershell
cd dashboard
npm install
npm run dev
```

Open http://localhost:5173

The Vite dev server proxies `/api` to the Express app on port 4000. The API reads `churn_db.customers`, `churn_db.predictions`, and `churn-basic/evaluation_summary.json`.

## Pages

| Route | Purpose |
|---|---|
| `/` | Totals, churn rate, high-risk count, contract churn |
| `/explore` | Filter customers by contract, tenure, and churn |
| `/performance` | Logistic Regression vs Random Forest test metrics |
| `/predictions` | Search customer ID, sort probability, filter risk |
| `/about` | Dataset, stack, model explanation, workflow |
