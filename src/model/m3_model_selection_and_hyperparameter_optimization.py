import os
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist
from tpot import TPOTClassifier
import logging

# Configure logging
logging.basicConfig(filename="../../ml_reports/model_selection_hyperparameter_reports/automl_results.log", level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Create reports directory
model_hyper_param_reports_dir = "../../ml_reports/model_selection_hyperparameter_reports"
os.makedirs(model_hyper_param_reports_dir, exist_ok=True)


def log_results(message):
    """Log messages to both console and file."""
    print(message)
    logging.info(message)


def save_html_report(df, filename):
    """Save pandas DataFrame as an HTML report with styling."""
    filepath = os.path.join(model_hyper_param_reports_dir, filename)

    # Identify the best model (highest accuracy)
    best_index = df['Accuracy'].idxmax()
    df.loc[best_index, 'Highlight'] = 'background-color: gold;'

    styled_df = df.style.apply(lambda row: [row.Highlight if col == 'Accuracy' else '' for col in df.columns], axis=1)

    html_content = f"""
    <html>
    <head>
        <style>
            table {{ width: 50%; border-collapse: collapse; margin: 20px 0; font-size: 18px; text-align: left; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; }}
            th {{ background-color: #f4b400; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
        </style>
    </head>
    <body>
        <h2>AutoML Model Selection Results</h2>
        {styled_df.render()}
    </body>
    </html>
    """

    with open(filepath, 'w') as f:
        f.write(html_content)

    log_results(f"Report saved at: {filepath}")


def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test, y_train, y_test = X_train[:1000], X_test[:300], y_train[:1000], y_test[:300]

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_xgboost(X_train, y_train, X_val, y_val):
    log_results("Running XGBoost Model...")
    model = xgb.XGBClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_val, model.predict(X_val))
    log_results(f"XGBoost Model Accuracy: {accuracy:.4f}")
    return {"Model": "XGBoost", "Accuracy": accuracy}


def run_tpot(X_train, y_train, X_val, y_val, run_id):
    log_results(f"Running TPOT AutoML Run {run_id}...")
    tpot = TPOTClassifier(generations=3, population_size=10, random_state=42 + run_id, max_time_mins=30, n_jobs=1)
    x_train_small, y_train_small = X_train[:1000], y_train[:1000]
    tpot.fit(x_train_small, y_train_small)
    accuracy = tpot.fitted_pipeline_.score(X_val, y_val)
    log_results(f"TPOT Run {run_id} - Best Model Accuracy: {accuracy:.4f}")
    return {"Model": f"TPOT Run {run_id}", "Accuracy": accuracy}


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    # Run XGBoost
    xgb_result = run_xgboost(X_train, y_train, X_val, y_val)

    # Run TPOT three times
    tpot_results = [run_tpot(X_train, y_train, X_val, y_val, i + 1) for i in range(3)]

    # Combine results and save
    all_results = [xgb_result] + tpot_results
    results_df = pd.DataFrame(all_results)
    save_html_report(results_df, "automl_results.html")

    log_results("All experiments completed successfully.")


if __name__ == "__main__":
    main()
