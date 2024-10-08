from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
from sklearn.impute import KNNImputer
import datetime

app = Flask(__name__)

# Load model, encoders, and scaler
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    model_top = pickle.load(f)

# Define feature mapping and top features
top_features = ['Vendor Name Encoded', 'CVE ID Encoded', 'CWE Encoded', 'Composite Risk Score']

weights = {
    'Base Score': 0.20,
    'Exploitability Score': 0.25,
    'Impact Score': 0.20,
    'Age': 0.10,
    'Update Lag': 0.10
}

# Helper function to safely encode categorical data
def safe_label_encode(encoder, value, default=-1):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return default

def preprocess_data_csv(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Strip whitespace
    df['Description'] = df['Description'].str.strip()
    df['Vendor Name'] = df['Vendor Name'].str.strip()

    # Replace invalid categories with NaN
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['Vulnerability Category'] = df['Vulnerability Category'].apply(lambda x: np.nan if isinstance(x, str) and re.search(url_pattern, x) else x)

    df['CWE'] = df['CWE'].apply(lambda x: np.nan if isinstance(x, str) and re.search(url_pattern, x) else x)

    # Handle date formats and NaN values
    df['Publish Date'] = pd.to_datetime(df['Publish Date'], errors='coerce')
    df['Update Date'] = pd.to_datetime(df['Update Date'], errors='coerce')
    
    df['EPSS Score'] = df['EPSS Score'].str.rstrip('%').astype('float') / 100.0

    # Handle numeric columns
    numeric_columns = [
        'Max CVSS Base Score',
        'Exploitability Score',
        'Impact Score',
        'Base Score',
        'EPSS Score'
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Impute missing values in numeric columns
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    return df

def preprocess_data_manual(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Strip whitespace
    df['Vendor Name'] = df['Vendor Name'].str.strip()

    
    # Remove URLs from 'CWE'
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['CWE'] = df['CWE'].apply(lambda x: np.nan if isinstance(x, str) and re.search(url_pattern, x) else x)

    # Handle date formats
    df['Publish Date'] = pd.to_datetime(df['Publish Date'], errors='coerce')
    df['Update Date'] = pd.to_datetime(df['Update Date'], errors='coerce')

    # Calculate Age and Update Lag
    df['Age'] = (pd.to_datetime('today') - df['Publish Date']).dt.days
    df['Update Lag'] = (df['Update Date'] - df['Publish Date']).dt.days

    # Fill NaNs in Age and Update Lag columns
    df['Age'] = df['Age'].fillna(0)
    df['Update Lag'] = df['Update Lag'].fillna(0)
    
    # Handle numeric columns
    numeric_columns = [
        'Exploitability Score',
        'Impact Score',
        'Base Score'
       # 'EPSS Score'
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Impute missing values in numeric columns
    imputer = KNNImputer(n_neighbors=5)
    df[ numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    return df

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_method = request.form.get('input_method')

    if input_method == 'manual':
        user_data_list = []

        for i in range(1000): # assuming max 1000 data sets
            cve_id = request.form.get(f'cve_id_{i}')
            vendor_name = request.form.get(f'vendor_name_{i}')
            cwe = request.form.get(f'cwe_{i}')
            base_score = request.form.get(f'base_score_{i}')
            exploitability_score = request.form.get(f'exploitability_score_{i}')
            impact_score = request.form.get(f'impact_score_{i}')
            publish_date = request.form.get(f'publish_date_{i}')
            update_date = request.form.get(f'update_date_{i}')

            if cve_id is None or vendor_name is None or cwe is None or base_score is None or exploitability_score is None or impact_score is None or publish_date is None or update_date is None:
                break

            # Create a DataFrame for input
            data = {
                'CVE ID': [cve_id],
                'Vendor Name': [vendor_name],
                'CWE': [cwe],
                'Base Score': [float(base_score)],
                'Exploitability Score': [float(exploitability_score)],
                'Impact Score': [float(impact_score)],
                'Publish Date': [publish_date],
                'Update Date': [update_date]
            }
            df = pd.DataFrame(data)

            # Preprocess the DataFrame
            df = preprocess_data_manual(df)

            # Calculate Age and Update Lag
            df['Age'] = (pd.to_datetime('today') - pd.to_datetime(df['Publish Date'])).dt.days
            df['Update Lag'] = (pd.to_datetime(df['Update Date']) - pd.to_datetime(df['Publish Date'])).dt.days

            # Calculate Composite Risk Score
            df['Composite Risk Score'] = df[['Base Score', 'Exploitability Score', 'Impact Score', 'Age', 'Update Lag']].apply(
                lambda x: sum(weights[col] * x[col] for col in weights.keys()), axis=1
            )

            # Normalize Composite Risk Score
            df[['Composite Risk Score']] = scaler.transform(df[['Composite Risk Score']])

            # Encode categorical columns
            df['Vendor Name Encoded'] = df['Vendor Name'].apply(lambda x: safe_label_encode(label_encoders['Vendor Name'], x))
            df['CVE ID Encoded'] = df['CVE ID'].apply(lambda x: safe_label_encode(label_encoders['CVE ID'], x))
            df['CWE Encoded'] = df['CWE'].apply(lambda x: safe_label_encode(label_encoders['CWE'], x))

            # Prepare features for prediction
            X_new = df[top_features].values

            # Predict EPSS score using the model
            df['Predicted EPSS Score'] = model_top.predict(X_new)

            # Rank vulnerabilities based on predicted EPSS score
            df['Rank'] = df['Predicted EPSS Score'].rank(ascending=False, method='first')  # Change method to 'first'
            df = df.sort_values(by='Rank').reset_index(drop=True)
            df['Rank'] = df.index + 1  # This will give a unique rank to each entry


            # Risk assessment based on EPSS score
            def risk_assessment(likelihood):
                if likelihood >= 0.9:
                    return 'Critical', 'Immediate Patching', '24 hours'
                elif 0.7 <= likelihood < 0.9:
                    return 'High', 'Priority Patching', '48 hours'
                elif 0.4 <= likelihood < 0.7:
                    return 'Medium', 'Scheduled Patching', '7 days'
                else:
                    return 'Low', 'Monitor', '30 days'

            df[['Risk Level', 'Suggested Action', 'Timeline']] = df['Predicted EPSS Score'].apply(risk_assessment).apply(pd.Series)

            # Keep only specified columns
            df = df[['CVE ID', 'Vendor Name', 'CWE', 'Composite Risk Score', 'Predicted EPSS Score', 'Rank', 'Risk Level', 'Suggested Action', 'Timeline']]

            # Add data to the list
            user_data_list.append(df)

        # Combine all user data into a single DataFrame
        combined_df = pd.concat(user_data_list, ignore_index=True)

        # Render results on a new page
        return render_template('results.html', results=combined_df.to_dict(orient='records'))

    elif input_method == 'csv':
        file = request.files.get('csv_file')
        if file is None:
            return "No file uploaded", 400

        # Read CSV data
        df = pd.read_csv(file)
        
        # Preprocess the CSV data
        df = preprocess_data_csv(df)

        # Calculate Age and Update Lag
        df['Age'] = (pd.to_datetime('today') - pd.to_datetime(df['Publish Date'])).dt.days
        df['Update Lag'] = (pd.to_datetime(df['Update Date']) - pd.to_datetime(df['Publish Date'])).dt.days

        # Calculate Composite Risk Score
        df['Composite Risk Score'] = df[['Base Score', 'Exploitability Score', 'Impact Score', 'Age', 'Update Lag']].apply(
            lambda x: sum(weights[col] * x[col] for col in weights.keys()), axis=1
        )

        # Normalize Composite Risk Score
        df[['Composite Risk Score']] = scaler.transform(df[['Composite Risk Score']])

        # Encode categorical columns
        df['Vendor Name Encoded'] = df['Vendor Name'].apply(lambda x: safe_label_encode(label_encoders['Vendor Name'], x))
        df['CVE ID Encoded'] = df['CVE ID'].apply(lambda x: safe_label_encode(label_encoders['CVE ID'], x))
        df['CWE Encoded'] = df['CWE'].apply(lambda x: safe_label_encode(label_encoders['CWE'], x))

        # Prepare features for prediction
        X_new = df[top_features].values

        # Predict EPSS score using the model
        df['Predicted EPSS Score'] = model_top.predict(X_new)

        # Rank vulnerabilities based on predicted EPSS score
        df['Rank'] = df['Predicted EPSS Score'].rank(ascending=False, method='first')  # Change method to 'first'
        df = df.sort_values(by='Rank').reset_index(drop=True)
        df['Rank'] = df.index + 1  # This will give a unique rank to each entry

        # Risk assessment based on EPSS score
        def risk_assessment(likelihood):
            if likelihood >= 0.35:
                return 'Critical', 'Immediate Patching', '24 hours'
            elif 0.25 <= likelihood < 0.35:
                return 'High', 'Priority Patching', '48 hours'
            elif 0.15 <= likelihood < 0.25:
                return 'Medium', 'Scheduled Patching', '7 days'
            else:
                return 'Low', 'Monitor', '30 days'


        df[['Risk Level', 'Suggested Action', 'Timeline']] = df['Predicted EPSS Score'].apply(risk_assessment).apply(pd.Series)

        # Keep only specified columns
        df = df[['CVE ID', 'Vendor Name', 'CWE', 'Composite Risk Score', 'Predicted EPSS Score', 'Rank', 'Risk Level', 'Suggested Action', 'Timeline']]

        # Render results on a new page
        return render_template('results.html', results=df.to_dict(orient='records'))

    else:
        return "Invalid input method", 400
    
if __name__ == '__main__':
    app.run(debug=True)