import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from xgboost import XGBClassifier

print("="*50)
print("STROKE PREDICTION MODEL TRAINING")
print("="*50)

# 1. Load data
print("\n[1/8] Loading dataset...")
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df.drop('id', axis=1)
print(f"✅ Loaded {len(df)} records")

# 2. Handle missing values
print("\n[2/8] Handling missing values...")
missing_before = df['bmi'].isnull().sum()
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
print(f"✅ Filled {missing_before} missing BMI values with median")

# 3. Split features and target
print("\n[3/8] Preparing features...")
X = df.drop('stroke', axis=1)
y = df['stroke']
print(f"✅ Features: {list(X.columns)}")
print(f"✅ Stroke cases: {y.sum()} out of {len(y)}")

# 4. Define column types
num_cols = ['age', 'avg_glucose_level', 'bmi']
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
binary_cols = ['hypertension', 'heart_disease']

# 5. Encode categorical
print("\n[4/8] Encoding categorical variables...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(X[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
print(f"✅ Created {len(encoded_df.columns)} encoded features")

# 6. Scale numerical
print("\n[5/8] Scaling numerical features...")
scaler = StandardScaler()
scaled = scaler.fit_transform(X[num_cols])
scaled_df = pd.DataFrame(scaled, columns=num_cols)
print("✅ Scaled age, avg_glucose_level, bmi")

# 7. Combine all features
X_final = pd.concat([scaled_df, X[binary_cols], encoded_df], axis=1)
feature_names = X_final.columns.tolist()

# 8. Train model
print("\n[6/8] Training Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, stratify=y, random_state=42
)

# ADD SMOTE TO BALANCE THE DATA
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {len(X_train)} training samples (balanced)")

# XGBoost with scale_pos_weight for imbalance
scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Model trained!")
# 9. Evaluate
print("\n[7/8] Evaluating model...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Save files
print("\n[8/8] Saving model files...")
joblib.dump(model, 'stroke_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("\n" + "="*50)
print("✅ ALL DONE! 4 files saved:")
print("="*50)
print("  • stroke_model.pkl")
print("  • scaler.pkl")
print("  • encoder.pkl")
print("  • feature_names.pkl")
print("\nYou can now upload these to your GitHub repo for deployment.")