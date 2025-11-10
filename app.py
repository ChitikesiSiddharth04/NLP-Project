"""
Bilingual Movie Review Sentiment Analysis Web Application
Supports both English and Telugu languages

This application demonstrates Natural Language Processing (NLP) techniques:
- Text Preprocessing for both English and Telugu
- Tokenization and stopword removal
- Feature Extraction (Bag of Words model)
- Text Classification (Naive Bayes for sentiment analysis)

NLP Libraries Used:
- NLTK: For English tokenization, stopwords, and stemming
- scikit-learn: For feature extraction (CountVectorizer) and classification
- Custom Telugu preprocessing pipeline
"""

import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load models and vectorizers
try:
    # English models
    english_model = pickle.load(open('english/model1.pkl', 'rb'))
    english_cv = pickle.load(open('english/bow.pkl', 'rb'))
    
    # Telugu models
    telugu_model = pickle.load(open('telugu/telugu_model.pkl', 'rb'))
    telugu_cv = pickle.load(open('telugu/telugu_vectorizer.pkl', 'rb'))
    
    print("All models loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Model files not found - {e}")
    print("Please run both MRA.py and telugu_MRA.py first to train the models.")
    exit(1)

# English NLP Preprocessing Pipeline Functions
def clean_english(text):
    """NLP Technique: Text Cleaning - Removes HTML tags using regex"""
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

def is_special_english(text):
    """NLP Technique: Text Normalization - Replaces special characters with spaces"""
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def to_lower_english(text):
    """NLP Technique: Case Normalization - Converts text to lowercase"""
    return text.lower()

def rem_stopwords_english(text):
    """NLP Technique: Stopword Removal - Removes common words using NLTK"""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)  # Tokenization: Breaking text into words
    return [w for w in words if w not in stop_words]

def stem_txt_english(text):
    """NLP Technique: Stemming - Reduces words to root forms using Snowball Stemmer"""
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

def preprocess_english_review(review):
    """
    English NLP Preprocessing Pipeline: Chains all NLP preprocessing techniques
    Steps: HTML cleaning -> Special char handling -> Lowercase -> Stopword removal -> Stemming
    """
    f1 = clean_english(review)              # Step 1: Remove HTML tags
    f2 = is_special_english(f1)             # Step 2: Handle special characters
    f3 = to_lower_english(f2)               # Step 3: Convert to lowercase
    f4 = rem_stopwords_english(f3)          # Step 4: Remove stopwords (tokenization happens here)
    return stem_txt_english(f4)             # Step 5: Apply stemming

# Telugu NLP Preprocessing Functions
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
    'అయిన', 'అయితే', 'కూడా', 'మాత్రం', 'అయితే', 'అయినా', 'అయినప్పటికీ'
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

# Routes
@app.route('/')
def home():
    return render_template('index.html', lang='english')

@app.route('/english')
def english():
    return render_template('index.html', lang='english')

@app.route('/telugu')
def telugu():
    return render_template('index.html', lang='telugu')

# English Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """English NLP Sentiment Analysis Endpoint: Performs end-to-end NLP pipeline"""
    if request.method == 'POST':
        review = request.form['review']
        
        if not review.strip():
            return jsonify({'error': 'Please enter a review.'}), 400
        
        try:
            # English NLP Pipeline: Preprocess -> Feature Extraction -> Classification
            processed_review = preprocess_english_review(review)  # English NLP Preprocessing
            review_vector = english_cv.transform([processed_review]).toarray()  # NLP Feature Extraction (BoW)
            y_pred = english_model.predict(review_vector)  # English NLP Text Classification
            
            sentiment = 'Positive' if y_pred[0] == 1 else 'Negative'
            
            # Return a JSON response
            return jsonify({
                'review': review, 
                'sentiment': sentiment,
                'processed_review': processed_review
            })
        except Exception as e:
            return jsonify({'error': f'Analysis error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid request method'}), 405

# Telugu Prediction route
@app.route('/predict_telugu', methods=['POST'])
def predict_telugu():
    """Telugu Sentiment Analysis Endpoint"""
    if request.method == 'POST':
        review = request.form['review']
        
        if not review.strip():
            return jsonify({'error': 'దయచేసి సమీక్షను నమోదు చేయండి'}), 400
        
        try:
            # Telugu NLP Pipeline: Preprocess -> Feature Extraction -> Classification
            processed_review = preprocess_telugu_review(review)  # Telugu NLP Preprocessing
            review_vector = telugu_cv.transform([processed_review]).toarray()  # Feature Extraction (BoW)
            y_pred = telugu_model.predict(review_vector)  # Telugu Text Classification
            
            sentiment = 'Positive' if y_pred[0] == 1 else 'Negative'
            sentiment_telugu = 'సానుకూలం' if y_pred[0] == 1 else 'ప్రతికూలం'
            
            # Return a JSON response with both English and Telugu results
            return jsonify({
                'review': review, 
                'sentiment': sentiment,
                'sentiment_telugu': sentiment_telugu,
                'processed_review': processed_review
            })
        except Exception as e:
            return jsonify({'error': f'విశ్లేషణలో లోపం: {str(e)}'}), 500
    
    return jsonify({'error': 'చెల్లని అభ్యర్థన పద్ధతి'}), 405

if __name__ == '__main__':
    app.run(debug=True, port=5001)