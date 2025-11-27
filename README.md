 Fake News Detection using Machine Learning (NLP)

This project applies Natural Language Processing and Machine Learning techniques to classify news text as either real or fake.  
The goal is to demonstrate a complete NLP pipeline using TF-IDF vectorization and a Logistic Regression classifier.



 Project Overview

The Fake News Detection system follows a standard NLP workflow:

- Loading and cleaning text data  
- Converting text into numerical vectors using TF-IDF  
- Training a machine learning classifier  
- Evaluating accuracy and performance metrics  
- Saving the trained model and vectorizer for reuse  

This project is suitable for students and developers who want to learn NLP fundamentals and text classification.



 Project Structure


Fake-News-Detection/
│── dataset/
│     └── fake_news.csv
│── src/
│     ├── fake_news_model.pkl
│     └── tfidf_vectorizer.pkl
│── notebook.ipynb
│── requirements.txt
│── README.md


 Machine Learning Workflow

 1. Data Preparation
- Loaded dataset using pandas  
- Removed missing values  
- Reset index for clean processing  
- Defined features (text) and labels (real/fake)

 2. Text Vectorization
- Applied TF-IDF Vectorizer  
- Removed English stopwords  
- Limited vocabulary size for efficiency  
- Transformed text into numerical feature vectors

 3. Model Training
Algorithm used:
- Logistic Regression  
  - Performs well on small and medium-sized text datasets  
  - Interpretable model  
  - Works effectively with TF-IDF features

 4. Model Evaluation
Evaluated using:
- Accuracy score  
- Classification report (precision, recall, f1-score)  
- Confusion matrix  
- Cross-validation for more reliable performance

 5. Model Saving
Saved essential components using joblib:
- Trained ML model  
- TF-IDF vectorizer

These files can be reused for deployment or future inference.

 Example Prediction

Input:
"Government launches new space mission tomorrow"

Output:
real

 How to Run the Project

 1. Clone the repository
 bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection


 2. Install dependencies
bash
pip install -r requirements.txt


 3. Open the notebook
bash
jupyter notebook


Run all cells to train, test, and save the model.

 Future Improvements

- Add a larger labeled dataset for higher accuracy  
- Integrate additional models (Naive Bayes, SVM)  
- Add advanced NLP preprocessing (lemmatization, stemming)  
- Build a Streamlit-based web interface  
- Deploy the model using FastAPI or Flask  



 Author

Shantanu Bawane  
AI/ML Student and Project Developer  
