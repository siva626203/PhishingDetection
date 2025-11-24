"""
Train Phishing Detection Models using Kaggle Dataset
"""

import kagglehub
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from urllib.parse import urlparse

print("=" * 60)
print("PHISHING DETECTION MODEL TRAINING")
print("=" * 60)

# ==================== STEP 1: DOWNLOAD DATASET ====================
print("\n[1/4] Downloading Kaggle dataset...")
try:
    path = kagglehub.dataset_download("ethancratchley/email-phishing-dataset")
    print(f"✓ Dataset downloaded to: {path}")
except Exception as e:
    print(f"✗ Error downloading dataset: {e}")
    print("Make sure you have Kaggle API credentials configured")
    exit(1)

# ==================== STEP 2: LOAD AND EXPLORE DATA ====================
print("\n[2/4] Loading and exploring data...")

# Find CSV files in the downloaded dataset
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"Found CSV files: {csv_files}")

if not csv_files:
    print("✗ No CSV files found in dataset")
    exit(1)

# Load the first CSV file
data_file = os.path.join(path, csv_files[0])
df = pd.read_csv(data_file)

print(f"\n✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# ==================== STEP 3: FEATURE EXTRACTION ====================
print("\n[3/4] Extracting features...")

def extract_features(row):
    """Extract features from email data"""
    features = {}
    
    # URL features
    features['has_suspicious_domain'] = 0
    features['url_length'] = 0
    features['has_ip_address'] = 0
    features['has_shortener'] = 0
    features['has_ssl'] = 0
    
    # Try different column names for URLs
    url_column = None
    for col in ['url', 'URL', 'link', 'Link', 'email_url']:
        if col in df.columns:
            url_column = col
            break
    
    if url_column and pd.notna(row.get(url_column)):
        url = str(row[url_column])
        features['url_length'] = len(url)
        
        if re.match(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            features['has_ip_address'] = 1
        
        shorteners = ['bit.ly', 'tinyurl', 'ow.ly', 'short.link']
        if any(s in url.lower() for s in shorteners):
            features['has_shortener'] = 1
        
        if 'https' in url:
            features['has_ssl'] = 1
        
        domain = urlparse(url).netloc.lower()
        if len(domain) > 50 or '-' in domain or domain.count('.') > 2:
            features['has_suspicious_domain'] = 1
    
    # Text features
    features['has_urgent_language'] = 0
    features['has_suspicious_keywords'] = 0
    features['suspicious_sender_format'] = 0
    features['text_length'] = 0
    features['has_numbers_in_sender'] = 0
    
    # Try different column names for content
    text_column = None
    for col in ['content', 'body', 'text', 'message', 'email_body']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column and pd.notna(row.get(text_column)):
        text = str(row[text_column]).lower()
        features['text_length'] = len(text)
        
        urgent_words = ['urgent', 'immediate', 'action required', 'verify now', 'confirm']
        if any(word in text for word in urgent_words):
            features['has_urgent_language'] = 1
        
        suspicious_keywords = ['verify', 'confirm', 'urgent', 'action required', 'update account']
        keyword_count = sum(1 for kw in suspicious_keywords if kw in text)
        if keyword_count > 2:
            features['has_suspicious_keywords'] = 1
    
    # Sender features
    sender_column = None
    for col in ['sender', 'from', 'email', 'sender_email']:
        if col in df.columns:
            sender_column = col
            break
    
    if sender_column and pd.notna(row.get(sender_column)):
        sender = str(row[sender_column]).lower()
        
        if re.search(r'\d', sender):
            features['has_numbers_in_sender'] = 1
        
        if '@' in sender:
            domain = sender.split('@')[1]
            if len(domain) > 30 or domain.count('.') > 2:
                features['suspicious_sender_format'] = 1
    
    return features

# Extract features for all rows
print("Extracting features from all emails...")
feature_list = []
for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"  Processed {idx}/{len(df)} rows...")
    feature_list.append(extract_features(row))

X = pd.DataFrame(feature_list)
print(f"✓ Extracted {len(X)} feature vectors with {len(X.columns)} features")

# ==================== STEP 4: PREPARE LABELS ====================
print("\nPreparing labels...")

# Find label column
label_column = None
for col in ['label', 'class', 'phishing', 'is_phishing', 'classification', 'result']:
    if col in df.columns:
        label_column = col
        break

if label_column is None:
    print("✗ Could not find label column in dataset")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

y = df[label_column].copy()

# Convert to binary (0 = legitimate, 1 = phishing)
if y.dtype == 'object':
    y = y.str.lower()
    y = y.map({'phishing': 1, 'legitimate': 0, 'spam': 1, 'ham': 0})

y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

print(f"✓ Labels prepared:")
print(f"  - Legitimate (0): {sum(y == 0)}")
print(f"  - Phishing (1): {sum(y == 1)}")

# ==================== STEP 5: TRAIN MODELS ====================
print("\n[4/4] Training models...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"✓ Random Forest trained")
print(f"  - Accuracy: {rf_accuracy:.4f}")
print(f"  - Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"  - Recall: {recall_score(y_test, rf_pred):.4f}")
print(f"  - F1-Score: {f1_score(y_test, rf_pred):.4f}")

# Train SVM
print("\nTraining SVM...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    probability=True,
    random_state=42,
    verbose=1
)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"✓ SVM trained")
print(f"  - Accuracy: {svm_accuracy:.4f}")
print(f"  - Precision: {precision_score(y_test, svm_pred):.4f}")
print(f"  - Recall: {recall_score(y_test, svm_pred):.4f}")
print(f"  - F1-Score: {f1_score(y_test, svm_pred):.4f}")

# ==================== STEP 6: SAVE MODELS ====================
print("\nSaving models...")

models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"✓ Created {models_dir} directory")

# Save models
with open(os.path.join(models_dir, 'rf_model.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)
    print(f"✓ Saved: {models_dir}/rf_model.pkl")

with open(os.path.join(models_dir, 'svm_model.pkl'), 'wb') as f:
    pickle.dump(svm_model, f)
    print(f"✓ Saved: {models_dir}/svm_model.pkl")

with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
    print(f"✓ Saved: {models_dir}/scaler.pkl")

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel Performance:")
print(f"  Random Forest Accuracy: {rf_accuracy:.2%}")
print(f"  SVM Accuracy: {svm_accuracy:.2%}")
print(f"\nModels saved in 'models/' directory")
print("\nYou can now run 'python app.py' to use trained models!")
print("=" * 60)