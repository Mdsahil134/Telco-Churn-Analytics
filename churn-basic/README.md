# Telecom Customer Churn Prediction (MongoDB + Classical ML)

## A. Project overview

This project predicts whether a telecom customer is likely to leave (churn) using two classical machine learning models: **Logistic Regression** and **Random Forest**.

Churn matters because acquiring a new customer usually costs more than keeping an existing one. If a company can identify high-risk customers early, it can offer better contracts, support, or pricing before those customers cancel.

Customer records are stored in MongoDB first. Training never reads the CSV directly. Predictions are written back to MongoDB so they can be queried like any other operational dataset.

## B. Dataset

- **Name:** IBM Telco Customer Churn (Kaggle: *Telco Customer Churn* by blastchar)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size verified in this run:** 7,043 rows, 21 columns
- **Target:** `Churn` (`Yes` / `No`)
- **Identifier:** `customerID` (kept for scoring, not used as a model feature)
- **Useful fields:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, and related service attributes

If you need to download the file yourself, search Kaggle for **Telco Customer Churn by blastchar**, or use the public IBM copy of the same table.

## C. Technologies used

- Python 3.13 (3.10 or later is required)
- MongoDB Community Server (local) or MongoDB Atlas via `MONGODB_URI`
- PyMongo
- pandas
- scikit-learn
- matplotlib

## D. Project architecture

```text
CSV dataset
    |
    v
MongoDB customers collection
    |
    v
Retrieve records into pandas
    |
    v
Clean data and explore churn patterns
    |
    v
Split data and preprocess features
    |
    v
Train Logistic Regression and Random Forest
    |
    v
Evaluate both models on held-out test data
    |
    v
Choose the model and refit on all customer records
    |
    v
Generate customer churn probabilities
    |
    v
MongoDB predictions collection
    |
    v
Query risk categories and highest-risk customers
```

Preprocessing is fitted on the **training split only** during evaluation. The selected model is later refit on all customers for operational scoring. Those full-dataset scores are **not** a second accuracy test.

## E. Installation and execution

MongoDB must be running before any script is executed.

### Windows PowerShell

```powershell
cd churn-basic
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python load_data.py
python train.py
python query_results.py
```

### Linux / macOS

```bash
cd churn-basic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python load_data.py
python train.py
python query_results.py
```

Run the scripts in that order: import → train → query.

Optional Atlas connection (do not commit credentials):

```powershell
$env:MONGODB_URI = "mongodb+srv://USER:PASSWORD@cluster.mongodb.net/"
```

The code uses `mongodb://localhost:27017/` when `MONGODB_URI` is not set.

## F. Actual model comparison

Held-out test set: 1,409 customers (20%), stratified so both splits keep a 26.5% churn rate. Metrics for Precision, Recall, and F1-score use the churn class (`1`).

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7388 | 0.5052 | 0.7834 | 0.6143 | 0.8417 |
| Random Forest | 0.7495 | 0.5183 | 0.7968 | 0.6280 | 0.8425 |

Confusion matrices on the same test set:

- Logistic Regression: TN=748, FP=287, FN=81, TP=293
- Random Forest: TN=758, FP=277, FN=76, TP=298

**Selected model: Random Forest**, because its test ROC-AUC (0.8425) is slightly higher than Logistic Regression (0.8417). Churn-class F1 is also higher (0.6280 vs 0.6143).

Accuracy is a weak headline number here. About 73.5% of customers did not churn, so a dummy model that always predicts “stay” would already look fairly accurate while catching **zero** churners.

## G. Feature importance

Top Random Forest features from the training-fitted pipeline (predictive contribution, **not causation**):

1. `tenure` — 0.176
2. `TotalCharges` — 0.126
3. `Contract_Two year` — 0.125
4. `MonthlyCharges` — 0.082
5. `InternetService_Fiber optic` — 0.080

Business reading of those associations:

- Longer tenure usually means the customer has already stayed; short tenure is a warning sign.
- Two-year contracts are a strong retention signal in this dataset.
- Fiber optic and higher monthly charges often appear with higher churn, which can reflect price sensitivity or service expectations.
- Electronic check (6th in the ranking) is a known billing-friction pattern in this table.

See `feature_importance.png` for the top 10 bars.

## H. Findings and limitations

Observed in the MongoDB data:

- Overall churn: **26.5%** (1,869 of 7,043).
- Month-to-month churn rate: **42.71%**, versus **11.27%** (one year) and **2.83%** (two year).
- Churned customers had shorter mean tenure (**18.0 vs 37.6 months**) and higher mean monthly charges (**74.44 vs 61.27**).

Model takeaway:

- Both models reached ROC-AUC around **0.84**, in the usual 0.80–0.85 range for this baseline.
- Random Forest was selected on test ROC-AUC.
- The models catch many churners (recall ~0.78–0.80) but also flag many loyal customers (precision ~0.51–0.52). That is a business trade-off: more outreach versus more wasted offers.

Prediction store from this run:

- 7,043 documents in `churn_db.predictions`
- High risk: 2,223; Medium: 1,861; Low: 2,959
- Average predicted probability: 0.4086

Risk bands (`>= 0.60` high, `>= 0.30` medium, otherwise low) are **illustrative**. A probability is not a guarantee that the customer will leave. Final scores include customers who were in the training set, so they must not be quoted as held-out accuracy.

Other limits: no hyperparameter search, no extra models, no causal claims, and `customerID` is excluded because it is an identifier, not a behaviour.

## Project files

| File | Purpose |
|---|---|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Real IBM Telco dataset (not generated) |
| `requirements.txt` | pymongo, pandas, scikit-learn, matplotlib |
| `load_data.py` | Validate CSV and insert into `churn_db.customers` |
| `train.py` | Read MongoDB, clean, train, evaluate, plot, save predictions |
| `query_results.py` | Aggregate risk counts and top customers from MongoDB |
| `feature_importance.png` | Top 10 Random Forest importances |
| `.gitignore` | Virtualenv, cache, and local env files |

## Optional React dashboard

The parent folder contains a React + Vite + Tailwind UI (`../dashboard`) and a Node/Express API (`../dashboard-api`) that read the same MongoDB collections. After this pipeline has been run, start the API then the UI as described in the repository `README.md`.
