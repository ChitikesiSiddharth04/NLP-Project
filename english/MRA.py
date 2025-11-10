"""
NLP Model Training Script for Sentiment Analysis

This script trains a machine learning model using various NLP techniques:
- Text Preprocessing: Cleaning, normalization, special character handling
- Tokenization: Breaking text into words using NLTK
- Stopword Removal: Removing common words that don't carry meaning
- Stemming: Reducing words to their root forms
- Feature Extraction: Bag of Words (BoW) model using CountVectorizer
- Classification: Naive Bayes classifiers (Gaussian, Multinomial, Bernoulli)

NLP Concepts Demonstrated:
1. Text preprocessing pipeline
2. Document vectorization (Bag of Words)
3. Feature extraction from text data
4. Text classification using probabilistic models
5. Model evaluation and comparison

Dataset: IMDB Movie Reviews (50,000 reviews)
Task: Binary sentiment classification (Positive/Negative)
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

data = pd.read_csv(os.path.join(script_dir, 'IMDB Dataset.csv'))
print(data.shape)
data.head()

data.info()

data.sentiment.value_counts()


data['sentiment'] = data['sentiment'].replace({'positive': 1, 'negative': 0})
data.head(10)

data.review[0]

def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)

data.review = data.review.apply(clean)
data.review[0]

def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

data.review = data.review.apply(is_special)
data.review[0]

def to_lower(text):
    return text.lower()

data.review = data.review.apply(to_lower)
data.review[0]

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

data.review = data.review.apply(rem_stopwords)
data.review[0]

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

data.review = data.review.apply(stem_txt)
data.review[0]

data.head()

X = np.array(data.iloc[:,0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(data.review).toarray()
print("X.shape = ",X.shape)
print("y.shape = ",y.shape)

print(X)

trainx,testx,trainy,testy = train_test_split(X,y,test_size=0.2,random_state=9)
print("Train shapes : X = {}, y = {}".format(trainx.shape,trainy.shape))
print("Test shapes : X = {}, y = {}".format(testx.shape,testy.shape))

gnb,mnb,bnb = GaussianNB(),MultinomialNB(alpha=1.0,fit_prior=True),BernoulliNB(alpha=1.0,fit_prior=True)
gnb.fit(trainx,trainy)
mnb.fit(trainx,trainy)
bnb.fit(trainx,trainy)

ypg = gnb.predict(testx)
ypm = mnb.predict(testx)
ypb = bnb.predict(testx)

# Calculate Accuracy Scores
print("="*60)
print("ACCURACY SCORES")
print("="*60)
print("Gaussian Naive Bayes Accuracy = ",accuracy_score(testy,ypg))
print("Multinomial Naive Bayes Accuracy = ",accuracy_score(testy,ypm))
print("Bernoulli Naive Bayes Accuracy = ",accuracy_score(testy,ypb))
print()

# Calculate F1 Scores
print("="*60)
print("F1 SCORES")
print("="*60)
f1_g = f1_score(testy, ypg, average='weighted')
f1_m = f1_score(testy, ypm, average='weighted')
f1_b = f1_score(testy, ypb, average='weighted')
print("Gaussian Naive Bayes F1 Score = ", f1_g)
print("Multinomial Naive Bayes F1 Score = ", f1_m)
print("Bernoulli Naive Bayes F1 Score = ", f1_b)
print()

# Calculate F1 Scores for each class
print("="*60)
print("F1 SCORES BY CLASS (Positive/Negative)")
print("="*60)
f1_g_class = f1_score(testy, ypg, average=None)
f1_m_class = f1_score(testy, ypm, average=None)
f1_b_class = f1_score(testy, ypb, average=None)
print("Gaussian Naive Bayes:")
print(f"  Negative (0): {f1_g_class[0]:.4f}")
print(f"  Positive (1): {f1_g_class[1]:.4f}")
print("Multinomial Naive Bayes:")
print(f"  Negative (0): {f1_m_class[0]:.4f}")
print(f"  Positive (1): {f1_m_class[1]:.4f}")
print("Bernoulli Naive Bayes:")
print(f"  Negative (0): {f1_b_class[0]:.4f}")
print(f"  Positive (1): {f1_b_class[1]:.4f}")
print()

# Confusion Matrices
print("="*60)
print("CONFUSION MATRICES")
print("="*60)

# Gaussian Naive Bayes Confusion Matrix
cm_g = confusion_matrix(testy, ypg)
print("\nGaussian Naive Bayes Confusion Matrix:")
print(cm_g)
print(f"True Negatives: {cm_g[0][0]}, False Positives: {cm_g[0][1]}")
print(f"False Negatives: {cm_g[1][0]}, True Positives: {cm_g[1][1]}")

# Multinomial Naive Bayes Confusion Matrix
cm_m = confusion_matrix(testy, ypm)
print("\nMultinomial Naive Bayes Confusion Matrix:")
print(cm_m)
print(f"True Negatives: {cm_m[0][0]}, False Positives: {cm_m[0][1]}")
print(f"False Negatives: {cm_m[1][0]}, True Positives: {cm_m[1][1]}")

# Bernoulli Naive Bayes Confusion Matrix (Best Model)
cm_b = confusion_matrix(testy, ypb)
print("\nBernoulli Naive Bayes Confusion Matrix (Selected Model):")
print(cm_b)
print(f"True Negatives: {cm_b[0][0]}, False Positives: {cm_b[0][1]}")
print(f"False Negatives: {cm_b[1][0]}, True Positives: {cm_b[1][1]}")
print()

# Detailed Classification Report for Best Model (Bernoulli)
print("="*60)
print("DETAILED CLASSIFICATION REPORT - BERNOULLI NAIVE BAYES")
print("="*60)
print(classification_report(testy, ypb, target_names=['Negative', 'Positive']))
print()

# Visualization of Confusion Matrix for Best Model
plt.figure(figsize=(10, 8))
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Bernoulli Naive Bayes (Best Model)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("Confusion matrix visualization saved as 'confusion_matrix.png'")
print()

# Save the best model (Bernoulli Naive Bayes)
pickle.dump(bnb,open(os.path.join(script_dir, 'model1.pkl'),'wb'))
pickle.dump(cv,open(os.path.join(script_dir, 'bow.pkl'),'wb'))
print("Model and vectorizer saved successfully!")
