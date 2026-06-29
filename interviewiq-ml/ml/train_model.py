# train_model.py
# ─────────────────────────────────────────────────────────────
# This script:
#   1. Loads the resume dataset (CSV)
#   2. Preprocesses the text
#   3. Trains a TF-IDF + Logistic Regression classifier
#   4. Evaluates accuracy
#   5. Saves the trained model as model.pkl
#
# Run this ONCE before starting the app:
#   python ml/train_model.py
# ─────────────────────────────────────────────────────────────

import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 50)
print("  InterviewIQ — Model Training")
print("=" * 50)

# ── Step 1: Load Dataset ──────────────────────────────────────
print("\n[1/4] Loading dataset...")
df = pd.read_csv('ml/resume_dataset.csv')
print(f"      Loaded {len(df)} resumes, {df['job_category'].nunique()} job categories")
print(f"      Categories: {list(df['job_category'].unique())}")

# ── Step 2: Prepare X (text) and y (label) ───────────────────
print("\n[2/4] Preparing training data...")
X = df['resume_text']
y = df['job_category']

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Training samples : {len(X_train)}")
print(f"      Testing samples  : {len(X_test)}")

# ── Step 3: Build Pipeline and Train ─────────────────────────
# Pipeline = TF-IDF vectorizer → Logistic Regression classifier
# TF-IDF converts text to numbers
# Logistic Regression learns which words map to which job category
print("\n[3/4] Training TF-IDF + Logistic Regression pipeline...")

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',   # Remove common words like "the", "is"
        ngram_range=(1, 2),     # Use single words AND pairs of words
        max_features=5000,      # Keep top 5000 most important words
        sublinear_tf=True       # Apply log scaling to term frequency
    )),
    ('clf', LogisticRegression(
        max_iter=1000,          # Maximum training iterations
        random_state=42,
        C=1.0                   # Regularization strength
    ))
])

model.fit(X_train, y_train)
print("      Training complete!")

# ── Step 4: Evaluate ─────────────────────────────────────────
print("\n[4/4] Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n      ✅ Test Accuracy: {accuracy * 100:.1f}%")
print("\n      Classification Report:")
print(classification_report(y_test, y_pred))

# ── Save the trained model ────────────────────────────────────
model_path = 'ml/model.pkl'
joblib.dump(model, model_path)
print(f"      ✅ Model saved to: {model_path}")

# ── Test with a sample resume ─────────────────────────────────
sample = "Python developer with Django Flask REST API experience. pandas numpy scikit-learn data analysis. PostgreSQL MySQL Git Docker."
predicted = model.predict([sample])[0]
probabilities = model.predict_proba([sample])[0]
classes = model.classes_

print(f"\n── Sample Prediction Test ──")
print(f"   Resume  : {sample[:60]}...")
print(f"   Predicted: {predicted}")
print(f"\n   All probabilities:")
for cls, prob in sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True):
    bar = '█' * int(prob * 20)
    print(f"   {cls:<22} {bar} {prob*100:.1f}%")

print("\n" + "=" * 50)
print("  Training Done! Run: python app.py")
print("=" * 50)
