"""
Telugu NLP Model Training Script for Sentiment Analysis

This script trains a machine learning model for Telugu movie review sentiment analysis:
- Text Preprocessing: Cleaning, normalization for Telugu text
- Tokenization: Breaking Telugu text into words
- Stopword Removal: Removing common Telugu words
- Feature Extraction: Bag of Words (BoW) model using CountVectorizer
- Classification: Naive Bayes classifiers for Telugu text

NLP Concepts Demonstrated:
1. Telugu text preprocessing pipeline
2. Document vectorization for Telugu
3. Feature extraction from Telugu text data
4. Text classification for Telugu language
5. Model evaluation and comparison

Dataset: Telugu Movie Reviews
Task: Binary sentiment classification (Positive/Negative)
"""

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

# Load Telugu dataset
data = pd.read_csv(os.path.join(script_dir, 'telugu_dataset.csv'))
print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

print("\nSentiment distribution:")
print(data.sentiment.value_counts())

# Convert sentiment to numeric (positive=1, negative=0)
data['sentiment'] = data['sentiment'].replace({'positive': 1, 'negative': 0})
print("\nAfter conversion:")
print(data.head())

# Telugu text preprocessing functions
def clean_telugu(text):
    """Clean Telugu text - remove HTML tags, special characters"""
    # Remove HTML tags
    cleaned = re.compile(r'<.*?>')
    text = re.sub(cleaned, '', text)
    
    # Keep only Telugu characters, spaces, and basic punctuation
    text = re.sub(r'[^\u0C00-\u0C7F\s।,!.?]', ' ', text)
    
    return text.strip()

def normalize_telugu(text):
    """Normalize Telugu text - handle spacing and punctuation"""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize punctuation
    text = re.sub(r'[!]+', '!', text)
    text = re.sub(r'[?]+', '?', text)
    text = re.sub(r'[.]+', '.', text)
    
    return text.strip()

# Common Telugu stopwords (basic set)
telugu_stopwords = {
    'ఈ', 'చాలా', 'ఉంది', 'ఉంది', 'చేసింది', 'కు', 'ను', 'ని', 'తో', 'వల్ల',
    'అయిన', 'అయితే', 'కూడా', 'మాత్రం', 'అయితే', 'అయినా', 'అయినప్పటికీ',
    'అయితే', 'అయినా', 'అయినప్పటికీ', 'అయితే', 'అయినా', 'అయినప్పటికీ'
}

def remove_telugu_stopwords(text):
    """Remove common Telugu stopwords"""
    words = text.split()
    filtered_words = [word for word in words if word not in telugu_stopwords and len(word) > 1]
    return ' '.join(filtered_words)

def preprocess_telugu_review(review):
    """
    Complete Telugu preprocessing pipeline
    """
    # Step 1: Clean Telugu text
    cleaned = clean_telugu(review)
    
    # Step 2: Normalize text
    normalized = normalize_telugu(cleaned)
    
    # Step 3: Convert to lowercase (for mixed text)
    lower_text = normalized.lower()
    
    # Step 4: Remove stopwords
    no_stopwords = remove_telugu_stopwords(lower_text)
    
    return no_stopwords

# Apply preprocessing to all reviews
print("\nApplying Telugu preprocessing...")
data['processed_review'] = data['review'].apply(preprocess_telugu_review)

print("\nSample processed reviews:")
for i in range(3):
    print(f"Original: {data['review'].iloc[i]}")
    print(f"Processed: {data['processed_review'].iloc[i]}")
    print("-" * 50)

# Prepare data for training
X = data['processed_review'].values
y = data['sentiment'].values

# Create CountVectorizer for Telugu text
cv = CountVectorizer(max_features=500, ngram_range=(1, 2))
X_vectorized = cv.fit_transform(X).toarray()

print(f"\nVectorized data shape: {X_vectorized.shape}")
print(f"Number of features: {len(cv.get_feature_names_out())}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train multiple models
gnb = GaussianNB()
mnb = MultinomialNB(alpha=1.0, fit_prior=True)
bnb = BernoulliNB(alpha=1.0, fit_prior=True)

print("\nTraining models...")
gnb.fit(X_train, y_train)
mnb.fit(X_train, y_train)
bnb.fit(X_train, y_train)

# Make predictions
ypg = gnb.predict(X_test)
ypm = mnb.predict(X_test)
ypb = bnb.predict(X_test)

# Calculate metrics
print("\n" + "="*60)
print("TELUGU MOVIE REVIEW SENTIMENT ANALYSIS RESULTS")
print("="*60)

print("\nACCURACY SCORES")
print("-" * 30)
print(f"Gaussian Naive Bayes Accuracy = {accuracy_score(y_test, ypg):.4f}")
print(f"Multinomial Naive Bayes Accuracy = {accuracy_score(y_test, ypm):.4f}")
print(f"Bernoulli Naive Bayes Accuracy = {accuracy_score(y_test, ypb):.4f}")

print("\nF1 SCORES")
print("-" * 30)
f1_g = f1_score(y_test, ypg, average='weighted')
f1_m = f1_score(y_test, ypm, average='weighted')
f1_b = f1_score(y_test, ypb, average='weighted')
print(f"Gaussian Naive Bayes F1 Score = {f1_g:.4f}")
print(f"Multinomial Naive Bayes F1 Score = {f1_m:.4f}")
print(f"Bernoulli Naive Bayes F1 Score = {f1_b:.4f}")

# Confusion Matrices
print("\nCONFUSION MATRICES")
print("-" * 30)

# Bernoulli Naive Bayes (usually best for text)
cm_b = confusion_matrix(y_test, ypb)
print("\nBernoulli Naive Bayes Confusion Matrix:")
print(cm_b)
print(f"True Negatives: {cm_b[0][0]}, False Positives: {cm_b[0][1]}")
print(f"False Negatives: {cm_b[1][0]}, True Positives: {cm_b[1][1]}")

# Detailed Classification Report for Best Model
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT - BERNOULLI NAIVE BAYES")
print("="*60)
print(classification_report(y_test, ypb, target_names=['Negative', 'Positive']))

# Visualization of Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Telugu Movie Review - Confusion Matrix (Bernoulli Naive Bayes)', 
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'telugu_confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("\nTelugu confusion matrix saved as 'telugu_confusion_matrix.png'")

# Save the best model and vectorizer
print("\nSaving model and vectorizer...")
with open(os.path.join(script_dir, 'telugu_model.pkl'), 'wb') as f:
    pickle.dump(bnb, f)

with open(os.path.join(script_dir, 'telugu_vectorizer.pkl'), 'wb') as f:
    pickle.dump(cv, f)

print("Model and vectorizer saved successfully!")

# Test with sample predictions
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

sample_reviews = [
    "ఈ సినిమా చాలా బాగుంది. నటీనటుల అభినయం అద్భుతంగా ఉంది.",
    "ఈ సినిమా చాలా చెడ్డగా ఉంది. కథ లేకుండా నడిపించారు.",
    "దర్శకుడు చాలా చక్కగా తెరకెక్కించాడు. ప్రతి సన్నివేశం అద్భుతంగా ఉంది."
]

for i, review in enumerate(sample_reviews, 1):
    processed = preprocess_telugu_review(review)
    vectorized = cv.transform([processed]).toarray()
    prediction = bnb.predict(vectorized)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    print(f"\nSample {i}:")
    print(f"Review: {review}")
    print(f"Processed: {processed}")
    print(f"Predicted: {sentiment}")

print("\nTelugu Movie Review Sentiment Analysis completed successfully!")