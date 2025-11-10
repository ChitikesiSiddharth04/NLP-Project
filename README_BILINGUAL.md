# 🎬 Bilingual Movie Review Sentiment Analysis

A comprehensive Natural Language Processing (NLP) application that performs sentiment analysis on movie reviews in both **English** and **Telugu** languages. This project demonstrates various NLP techniques including text preprocessing, tokenization, feature extraction, and machine learning classification.

## 🌟 Features

- **Bilingual Support**: Analyze movie reviews in both English and Telugu
- **Modern Web Interface**: Beautiful, responsive UI with language switching
- **Advanced NLP Pipeline**: Comprehensive text preprocessing for both languages
- **Machine Learning Models**: Multiple Naive Bayes classifiers for sentiment analysis
- **Real-time Analysis**: Instant sentiment prediction with detailed results
- **Interactive Sample Reviews**: Quick testing with pre-loaded examples

## 🛠️ Technical Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **NLP Libraries**: NLTK (English), scikit-learn (both languages)
- **Machine Learning**: Naive Bayes classifiers (Gaussian, Multinomial, Bernoulli)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 📁 Project Structure

```
Movie-Review-Analysis-main/
├── app.py                          # Original English-only Flask app
├── bilingual_app.py                # 🆕 Bilingual Flask application
├── MRA.py                          # English model training script
├── telugu_MRA.py                   # 🆕 Telugu model training script
├── telugu_dataset.csv              # 🆕 Telugu movie review dataset
├── model1.pkl                      # English trained model
├── bow.pkl                         # English vectorizer
├── telugu_model.pkl                # 🆕 Telugu trained model
├── telugu_vectorizer.pkl           # 🆕 Telugu vectorizer
├── telugu_confusion_matrix.png     # 🆕 Telugu model evaluation
├── confusion_matrix.png           # English model evaluation
├── requirements.txt                  # Python dependencies
├── templates/
│   ├── index.html                  # English web interface
│   └── telugu_index.html           # 🆕 Telugu web interface
└── README.md                       # This file
```

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Movie-Review-Analysis-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (for English processing):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Train the models**:
   ```bash
   # Train English model
   python MRA.py
   
   # Train Telugu model
   python telugu_MRA.py
   ```

5. **Run the bilingual application**:
   ```bash
   python bilingual_app.py
   ```

6. **Access the application**:
   - Open your browser and go to: `http://127.0.0.1:5001`
   - Switch between English and Telugu using the language selector

## 🎯 NLP Techniques Implemented

### English Language Processing:
- **Text Cleaning**: HTML tag removal using regex
- **Text Normalization**: Special character handling
- **Case Normalization**: Convert to lowercase
- **Tokenization**: Word-level tokenization using NLTK
- **Stopword Removal**: Filter common English words
- **Stemming**: Reduce words to root forms using Snowball Stemmer
- **Feature Extraction**: Bag of Words model using CountVectorizer

### Telugu Language Processing:
- **Text Cleaning**: Remove HTML tags and non-Telugu characters
- **Unicode Handling**: Preserve Telugu script (Unicode range: \u0C00-\u0C7F)
- **Text Normalization**: Handle spacing and punctuation
- **Stopword Removal**: Custom Telugu stopword dictionary
- **Feature Extraction**: Bag of Words model optimized for Telugu
- **Script Preservation**: Maintain Telugu character integrity

## 📊 Model Performance

### English Model Results:
- **Bernoulli Naive Bayes** (Best Model):
  - Accuracy: ~85-90%
  - F1-Score: Weighted average based on class distribution
  - Confusion Matrix: Available in `confusion_matrix.png`

### Telugu Model Results:
- **Bernoulli Naive Bayes** (Best Model):
  - Accuracy: ~95%+ (on test dataset)
  - F1-Score: Excellent performance on Telugu text
  - Confusion Matrix: Available in `telugu_confusion_matrix.png`

## 🌐 Usage Guide

### English Interface:
1. Navigate to `http://127.0.0.1:5001/english`
2. Enter your movie review in English
3. Click "Analyze" to get sentiment prediction
4. View results with confidence scores

### Telugu Interface:
1. Navigate to `http://127.0.0.1:5001/telugu`
2. Enter your movie review in Telugu script
3. Click "విశ్లేషించండి" (Analyze) to get sentiment prediction
4. View results in both Telugu and English

### Sample Reviews for Testing:

**English (Positive)**:
- "This movie was absolutely fantastic! The acting was superb and the story was engaging."
- "I loved every minute of this film. Great direction and excellent performances."

**English (Negative)**:
- "This movie was terrible. Poor acting and weak storyline."
- "Complete waste of time. The plot was confusing and the acting was awful."

**Telugu (సానుకూలం - Positive)**:
- "ఈ సినిమా చాలా బాగుంది. నటీనటుల అభినయం అద్భుతంగా ఉంది."
- "దర్శకుడు చాలా చక్కగా తెరకెక్కించాడు. ప్రతి సన్నివేశం అద్భుతంగా ఉంది."

**Telugu (ప్రతికూలం - Negative)**:
- "ఈ సినిమా చాలా చెడ్డగా ఉంది. కథ లేకుండా నడిపించారు."
- "సినిమా పూర్తిగా నిరాశపరిచింది. డబ్బు వృథా."

## 🔧 API Endpoints

### English Sentiment Analysis:
- **Endpoint**: `/predict`
- **Method**: POST
- **Parameters**: `review` (English text)
- **Response**: JSON with review, sentiment, and processed text

### Telugu Sentiment Analysis:
- **Endpoint**: `/predict_telugu`
- **Method**: POST
- **Parameters**: `review` (Telugu text)
- **Response**: JSON with review, sentiment (English & Telugu), and processed text

## 📈 Future Enhancements

- **Multi-language Support**: Extend to other Indian languages (Hindi, Tamil, etc.)
- **Deep Learning Models**: Implement LSTM, BERT for better accuracy
- **Real-time Data**: Integration with movie review APIs
- **Sentiment Intensity**: Add confidence scores and sentiment strength
- **Mobile App**: Develop mobile application for wider accessibility
- **Database Integration**: Store and analyze user reviews over time

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NLTK**: Natural Language Toolkit for English NLP
- **scikit-learn**: Machine learning library
- **Flask**: Web framework
- **Telugu Dataset**: Inspired by research on Telugu sentiment analysis
- **Open Source Community**: For continuous support and contributions

---

**Made with ❤️ for the love of Cinema and Natural Language Processing!**