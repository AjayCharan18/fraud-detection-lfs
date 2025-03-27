# fraud_detection.py

# ==================================================
# Category 1: Import Required Libraries
# ==================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================================================
# Category 2: Load and Combine .pkl Files
# ==================================================
def load_and_combine_pkl_files(data_folder='data'):
    """
    Loads and combines all .pkl files from the specified folder into a single DataFrame.
    """
    try:
        # Check if the data folder exists
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"The folder '{data_folder}' does not exist.")

        # Find all .pkl files in the data folder
        pkl_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
        
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in the folder '{data_folder}'.")

        # Load and combine all .pkl files into a single DataFrame
        df_list = []
        
        for pkl_file in pkl_files:
            file_path = os.path.join(data_folder, pkl_file)
            df = pd.read_pickle(file_path)
            df_list.append(df)

        # Combine all DataFrames into one
        combined_df = pd.concat(df_list, ignore_index=True)

        logging.info(f" Loaded {len(pkl_files)} .pkl files into a single DataFrame with {len(combined_df)} rows.")
        logging.info("Sample data:")
        logging.info(combined_df.head())

        return combined_df

    except Exception as e:
        logging.error(f" An error occurred while loading and combining .pkl files: {e}")
        raise

# ==================================================
# Category 3: Feature Engineering
# ==================================================
def feature_engineering(df):
    """
    Performs feature engineering on the dataset.
    """
    try:
        # Scenario 1: Transactions with amount > 220 are fraudulent
        df['HIGH_AMOUNT'] = df['TX_AMOUNT'].apply(lambda x: 1 if x > 220 else 0)

        # Scenario 2: Fraudulent terminals (terminals marked as fraudulent for 28 days)
        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        df['FRAUD_TERMINAL'] = df.groupby('TERMINAL_ID')['TX_FRAUD'].transform('max')

        # Scenario 3: Fraudulent customers (customers with 1/3 of transactions multiplied by 5)
        df['FRAUD_CUSTOMER'] = df.groupby('CUSTOMER_ID')['TX_FRAUD'].transform('max')

        # Additional features
        df['DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek
        df['HOUR_OF_DAY'] = df['TX_DATETIME'].dt.hour

        # Fraud count features
        df['FRAUD_COUNT_TERMINAL'] = df.groupby('TERMINAL_ID')['TX_FRAUD'].transform('sum')
        df['FRAUD_COUNT_CUSTOMER'] = df.groupby('CUSTOMER_ID')['TX_FRAUD'].transform('sum')

        # Customer spending habits
        df['CUSTOMER_MEAN_AMOUNT'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('mean')
        df['CUSTOMER_STD_AMOUNT'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('std').replace(0, 1)  # Avoid division by zero
        df['AMOUNT_TO_MEAN_RATIO'] = df['TX_AMOUNT'] / df['CUSTOMER_MEAN_AMOUNT']
        df['AMOUNT_TO_STD_RATIO'] = df['TX_AMOUNT'] / df['CUSTOMER_STD_AMOUNT']

        # Transaction frequency
        df['TRANSACTION_FREQUENCY'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].transform('count')

        logging.info(" Feature Engineering Applied.")
        logging.info("Sample data with new features:")
        logging.info(df.head())

        return df

    except Exception as e:
        logging.error(f" An error occurred during feature engineering: {e}")
        raise

# ==================================================
# Category 4: Select Features and Target
# ==================================================
def select_features_and_target(df):
    """
    Selects features and target variable for modeling.
    """
    try:
        features = [
            'TX_AMOUNT', 'HIGH_AMOUNT', 'FRAUD_TERMINAL', 'FRAUD_CUSTOMER',
            'DAY_OF_WEEK', 'HOUR_OF_DAY', 'FRAUD_COUNT_TERMINAL', 'FRAUD_COUNT_CUSTOMER',
            'CUSTOMER_MEAN_AMOUNT', 'CUSTOMER_STD_AMOUNT', 'AMOUNT_TO_MEAN_RATIO',
            'AMOUNT_TO_STD_RATIO', 'TRANSACTION_FREQUENCY'
        ]
        X = df[features]
        y = df['TX_FRAUD']

        # Retain CUSTOMER_ID and TERMINAL_ID for later use
        metadata = df[['CUSTOMER_ID', 'TERMINAL_ID']]

        logging.info(" Features and Target Selected.")
        logging.info(f"Features shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")

        return X, y, metadata

    except Exception as e:
        logging.error(f" An error occurred while selecting features and target: {e}")
        raise

# ==================================================
# Category 5: Split the Data into Training and Testing Sets
# ==================================================
def split_data(X, y, metadata):
    """
    Splits the data into training and testing sets.
    """
    try:
        X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
            X, y, metadata, test_size=0.3, random_state=42
        )

        logging.info(" Data Split Completed.")
        logging.info(f"Training set shape: {X_train.shape}, {y_train.shape}")
        logging.info(f"Testing set shape: {X_test.shape}, {y_test.shape}")

        return X_train, X_test, y_train, y_test, metadata_test

    except Exception as e:
        logging.error(f" An error occurred while splitting the data: {e}")
        raise

# ==================================================
# Category 6: Data Preprocessing
# ==================================================
def preprocess_data(X_train, X_test):
    """
    Preprocesses the data by standardizing the features.
    """
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert scaled arrays back to DataFrames with column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        logging.info(" Data Preprocessing Completed.")
        logging.info("Sample scaled training data:")
        logging.info(X_train_scaled.head())

        return X_train_scaled, X_test_scaled, scaler

    except Exception as e:
        logging.error(f" An error occurred during data preprocessing: {e}")
        raise

# ==================================================
# Category 7: Train Random Forest Model
# ==================================================
def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a Random Forest model.
    """
    try:
        # Handle class imbalance using class_weight
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        logging.info(" Random Forest Model Trained.")
        logging.info("Classification Report:")
        logging.info(classification_report(y_test, y_pred_rf))
        logging.info(f"Random Forest ROC AUC Score: {roc_auc_score(y_test, y_pred_rf)}")

        return rf_model, y_pred_rf

    except Exception as e:
        logging.error(f" An error occurred while training the Random Forest model: {e}")
        raise

# ==================================================
# Category 8: Train XGBoost Model
# ==================================================
def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates an XGBoost model.
    """
    try:
        # Handle class imbalance using scale_pos_weight
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)

        logging.info(" XGBoost Model Trained.")
        logging.info("Classification Report:")
        logging.info(classification_report(y_test, y_pred_xgb))
        logging.info(f"XGBoost ROC AUC Score: {roc_auc_score(y_test, y_pred_xgb)}")

        return xgb_model, y_pred_xgb

    except Exception as e:
        logging.error(f" An error occurred while training the XGBoost model: {e}")
        raise

# ==================================================
# Category 9: Visualize Results
# ==================================================
def visualize_results(y_test, y_pred_rf, y_pred_xgb):
    """
    Visualizes the results using confusion matrices and ROC curves.
    """
    try:
        # Confusion Matrix for Random Forest
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Confusion Matrix for XGBoost
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Greens')
        plt.title('XGBoost Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # ROC Curve for Random Forest
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_pred_rf):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        # ROC Curve for XGBoost
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_pred_xgb):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    except Exception as e:
        logging.error(f" An error occurred while visualizing results: {e}")
        raise

# ==================================================
# Category 10: Save Models and Predictions
# ==================================================
def save_models_and_predictions(rf_model, xgb_model, scaler, X_test, y_test, y_pred_rf, y_pred_xgb, metadata_test):
    """
    Saves the trained models, scaler, and predictions to disk.
    """
    try:
        # Save models and scaler
        joblib.dump(rf_model, 'random_forest_fraud.pkl')
        joblib.dump(xgb_model, 'xgboost_fraud.pkl')
        joblib.dump(scaler, 'scaler.pkl')

        # Save predictions
        predictions = pd.DataFrame({
            'TRANSACTION_ID': X_test.index,
            'CUSTOMER_ID': metadata_test['CUSTOMER_ID'],
            'TERMINAL_ID': metadata_test['TERMINAL_ID'],
            'TX_AMOUNT': X_test['TX_AMOUNT'],
            'TX_FRAUD (Actual)': y_test,
            'TX_FRAUD (Predicted - RF)': y_pred_rf,
            'TX_FRAUD (Predicted - XGB)': y_pred_xgb
        })
        predictions.to_csv('predictions.csv', index=False)

        logging.info(" Models, scaler, and predictions saved successfully.")

    except Exception as e:
        logging.error(f" An error occurred while saving models and predictions: {e}")
        raise

# ==================================================
# Category 11: Load and Use Models
# ==================================================
def load_and_predict():
    """
    Loads the saved models and makes predictions on new data.
    """
    try:
        rf_model = joblib.load('random_forest_fraud.pkl')
        xgb_model = joblib.load('xgboost_fraud.pkl')
        scaler = joblib.load('scaler.pkl')

        new_data = pd.DataFrame({
            'TX_AMOUNT': [150, 300],
            'HIGH_AMOUNT': [0, 1],
            'FRAUD_TERMINAL': [0, 1],
            'FRAUD_CUSTOMER': [0, 1],
            'DAY_OF_WEEK': [3, 5],
            'HOUR_OF_DAY': [10, 20],
            'FRAUD_COUNT_TERMINAL': [0, 2],
            'FRAUD_COUNT_CUSTOMER': [0, 1],
            'CUSTOMER_MEAN_AMOUNT': [100, 200],
            'CUSTOMER_STD_AMOUNT': [50, 100],
            'AMOUNT_TO_MEAN_RATIO': [1.5, 1.5],
            'AMOUNT_TO_STD_RATIO': [3.0, 3.0],
            'TRANSACTION_FREQUENCY': [10, 20]
        })

        # Scale the new data and convert back to DataFrame with feature names
        new_data_scaled = scaler.transform(new_data)
        new_data_scaled = pd.DataFrame(new_data_scaled, columns=new_data.columns)

        logging.info("\nRandom Forest Predictions: %s", rf_model.predict(new_data_scaled))
        logging.info("XGBoost Predictions: %s", xgb_model.predict(new_data_scaled))

    except Exception as e:
        logging.error(f" An error occurred while loading and predicting with models: {e}")
        raise

# ==================================================
# Category 12: Generate Summary Report
# ==================================================
def generate_summary_report(y_test, y_pred_rf, y_pred_xgb):
    """
    Generates a summary report similar to the provided screenshot.
    """
    total_transactions = len(y_test)
    detection_rate_rf = roc_auc_score(y_test, y_pred_rf) * 100
    detection_rate_xgb = roc_auc_score(y_test, y_pred_xgb) * 100

    failed_results_rf = (y_test != y_pred_rf).sum()
    failed_results_xgb = (y_test != y_pred_xgb).sum()

    failed_percentage_rf = (failed_results_rf / total_transactions) * 100
    failed_percentage_xgb = (failed_results_xgb / total_transactions) * 100

    print("# Data Management")
    print("## Data Management")
    print(f"- Total Transactions\n  **{total_transactions}**\n  Processed transactions\n")
    print("## Data Management")
    print(f"- Detection Rate (Random Forest)\n  **{detection_rate_rf:.1f}%**\n")
    print(f"- Detection Rate (XGBoost)\n  **{detection_rate_xgb:.1f}%**\n")
    print("## Data Management")
    print(f"- Model Training (Random Forest)\n  **{failed_percentage_rf:.2f}%**\n  Failed Results\n  **{failed_results_rf}**\n  {failed_percentage_rf:.2f}% of total\n")
    print(f"- Model Training (XGBoost)\n  **{failed_percentage_xgb:.2f}%**\n  Failed Results\n  **{failed_results_xgb}**\n  {failed_percentage_xgb:.2f}% of total\n")
    print("---\n")
    print("# Fraud Detection Performance")
    print("Model performance metrics over time\n")
    print("---\n")
    print("Performance chart would render here\n")
    print("(Connects to Python backend for real data)\n")

# ==================================================
# Main Execution
# ==================================================
if __name__ == "__main__":
    try:
        # Step 1: Load and combine .pkl files
        df = load_and_combine_pkl_files('data')

        # Step 2: Feature Engineering
        df = feature_engineering(df)

        # Step 3: Select Features and Target
        X, y, metadata = select_features_and_target(df)

        # Step 4: Split the Data
        X_train, X_test, y_train, y_test, metadata_test = split_data(X, y, metadata)

        # Step 5: Preprocess the Data
        X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

        # Step 6: Train Random Forest Model
        rf_model, y_pred_rf = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)

        # Step 7: Train XGBoost Model
        xgb_model, y_pred_xgb = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)

        # Step 8: Visualize Results
        visualize_results(y_test, y_pred_rf, y_pred_xgb)

        # Step 9: Save Models and Predictions
        save_models_and_predictions(rf_model, xgb_model, scaler, X_test, y_test, y_pred_rf, y_pred_xgb, metadata_test)

        # Step 10: Generate Summary Report
        generate_summary_report(y_test, y_pred_rf, y_pred_xgb)

        # Step 11: Load and Use Models
        load_and_predict()

    except Exception as e:
        logging.error(f" An error occurred in the main execution: {e}")