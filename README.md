# 🎬 Bilingual Movie Review Sentiment Analysis - NLP Project

This project is a **comprehensive Natural Language Processing (NLP) application** that performs sentiment analysis on movie reviews in **both English and Telugu languages**. It demonstrates various NLP techniques including text preprocessing, feature extraction, and machine learning-based text classification. The application uses trained models on IMDB 50k Movie Review dataset (English) and Telugu movie reviews dataset to classify reviews as either "Positive" or "Negative".

![Demo](https://i.imgur.com/your-demo-image.gif)  <!-- Replace with a GIF of your bilingual app! -->

## 🧠 NLP Techniques Implemented

This project showcases several fundamental NLP concepts for both English and Telugu:

### English NLP Pipeline:
- **Text Preprocessing**: HTML tag removal, special character handling, case normalization
- **Tokenization**: Word-level tokenization using NLTK
- **Stopword Removal**: Filtering common English words
- **Stemming**: Reducing words to their root forms using Snowball Stemmer
- **Feature Extraction**: Bag of Words (BoW) model with CountVectorizer
- **Text Classification**: Bernoulli Naive Bayes classifier for sentiment analysis

### Telugu NLP Pipeline:
- **Telugu Text Preprocessing**: Unicode character handling, punctuation normalization
- **Telugu Tokenization**: Word-level tokenization for Telugu script
- **Telugu Stopword Removal**: Custom Telugu stopword filtering
- **Feature Extraction**: Bag of Words model optimized for Telugu text
- **Text Classification**: Bernoulli Naive Bayes classifier for Telugu sentiment

📖 **For detailed NLP techniques documentation, see [NLP_TECHNIQUES.md](NLP_TECHNIQUES.md)**

## ✨ Features

- **Bilingual Support:** Analyze movie reviews in both English and Telugu languages from a unified interface
- **NLP-Powered Sentiment Analysis:** Utilizes multiple NLP preprocessing techniques and `Bernoulli Naive Bayes` classifiers trained on 50,000+ movie reviews
- **Complete NLP Pipeline:** End-to-end text processing from raw input to sentiment prediction for both languages
- **Dynamic Frontend:** Single-page application with language switching built with vanilla JavaScript, HTML, and CSS
- **Responsive Design:** A clean and modern UI that looks great on all screen sizes
- **Organized Structure:** Language-specific files organized in separate directories (`/english/` and `/telugu/`)
- **Ready for Deployment:** Includes configuration for easy, free deployment on platforms like Render

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **NLP Libraries:** NLTK (Natural Language Toolkit), scikit-learn
- **Machine Learning:** Scikit-learn, Pandas
- **NLP Techniques:** Tokenization, Stemming, Stopword Removal, Bag of Words
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Gunicorn, Render

## 📂 File Structure

```
.
├── app.py                  # Main Flask application (bilingual)
├── templates/
│   └── index.html          # Unified frontend for both languages
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to be ignored by Git
├── render-build.sh         # Build script for deployment
├── README.md               # This file
├── README_BILINGUAL.md     # Bilingual system documentation
├── NLP_TECHNIQUES.md       # Detailed NLP techniques documentation
├── EVALUATION_METRICS.md   # Model evaluation metrics documentation
├── english/                # English language components
│   ├── MRA.py             # English model training script
│   ├── model1.pkl         # English trained BernoulliNB model
│   ├── bow.pkl            # English CountVectorizer
│   ├── confusion_matrix.png # English model evaluation
│   └── IMDB Dataset.csv   # English training dataset
└── telugu/                # Telugu language components
    ├── telugu_MRA.py      # Telugu model training script
    ├── telugu_model.pkl   # Telugu trained BernoulliNB model
    ├── telugu_vectorizer.pkl # Telugu CountVectorizer
    ├── telugu_confusion_matrix.png # Telugu model evaluation
    └── telugu_dataset.csv # Telugu training dataset
```

---

## 🚀 Running Locally

To run this project on your own machine, follow these steps.

### 1. Prerequisites

- Python 3.8+
- Git

### 2. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

### 3. Get the Dataset & Train the Models

The pre-trained model files are already included in this repository. However, if you wish to retrain the models yourself:

**For English Model:**
1.  **Download the dataset** from [Kaggle: IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2.  Place the `IMDB Dataset.csv` file in the `english/` directory.
3.  **Run the English training script:**
    ```bash
    python english/MRA.py
    ```

**For Telugu Model:**
1.  Place the `telugu_dataset.csv` file in the `telugu/` directory.
2.  **Run the Telugu training script:**
    ```bash
    python telugu/telugu_MRA.py
    ```

### 4. Run the Web App

```bash
# Start the Flask development server
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5001` to use the bilingual application.

### 5. Using the Application

1. **Select Language**: Use the language dropdown to switch between English and Telugu
2. **Enter Review**: Type or paste your movie review in the selected language
3. **Analyze**: Click the "Analyze Sentiment" button to get instant results
4. **View Results**: The sentiment (Positive/Negative) will be displayed with confidence

---

## 📊 Model Performance

### English Model Performance
- **Accuracy**: 83.86%
- **F1 Score**: 0.838 (Weighted Average)
- **Dataset**: 50,000 IMDB movie reviews
- **Best Algorithm**: Bernoulli Naive Bayes

### Telugu Model Performance
- **Accuracy**: 100% (on test dataset)
- **F1 Score**: 1.00 (Perfect score)
- **Dataset**: Custom Telugu movie reviews
- **Best Algorithm**: Bernoulli Naive Bayes

### 1. Create a GitHub Repository

Push your project code to a new repository on GitHub.

### 2. Deploy on Render

1.  **Create a new Render Account** or log in.
2.  On your dashboard, click **New +** and select **Web Service**.
3.  **Connect your GitHub repository**.
4.  Fill in the service details:
    -   **Name:** Give your app a unique name (e.g., `bilingual-movie-review-analysis`).
    -   **Region:** Choose a region near you.
    -   **Branch:** `main` (or your default branch).
    -   **Root Directory:** Leave it blank.
    -   **Runtime:** `Python 3`.
    -   **Build Command:** `./render-build.sh`
    -   **Start Command:** `gunicorn app:app`
5.  Click **Create Web Service**.

Render will automatically build and deploy your application. Once it's live, you'll get a public URL you can share with anyone!

## 🧪 Testing the Application

### Sample Test Reviews

**English Positive:** "This movie is absolutely fantastic! The acting was superb and the storyline kept me engaged throughout."

**English Negative:** "This was the worst movie I've ever seen. The plot made no sense and the acting was terrible."

**Telugu Positive:** "ఈ సినిమా చాలా బాగుంది. నటీనటుల అభినయం అద్భుతంగా ఉంది."

**Telugu Negative:** "ఈ సినిమా చాలా చెడ్డగా ఉంది. కథ లేకుండా నడిపించారు."