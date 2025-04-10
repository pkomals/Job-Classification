import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

class JobClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,  # Reduced from 5000
                ngram_range=(1, 2),  # Keep bigrams for context
                min_df=3,  # Words must appear in at least 3 documents
                max_df=0.8,  # Words must appear in less than 80% of documents
                stop_words='english'
            )),
            ('clf', RandomForestClassifier(
                n_estimators=100,  # Reduced from 200
                max_depth=20,  # Reduced from 50
                min_samples_split=5,  # Added minimum samples for split
                min_samples_leaf=2,  # Added minimum samples in leaf
                class_weight='balanced',  # Handle class imbalance
                max_features='sqrt',  # Use sqrt of total features
                random_state=42
            ))
        ])
    
    def train(self, X, y):
        """Train the model with improved parameters and evaluation"""
        print("Training job classifier...")
        print("Vectorizing text data and training model...")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred))
        
        # Perform cross-validation
        print("\nPerforming cross-validation...")
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    
    def predict(self, text):
        """Make predictions with probability scores"""
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        return prediction, probabilities
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.pipeline = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
