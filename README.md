# 📰 Fake News Detection

This project aims to detect fake news from real news using **Natural Language Processing (NLP)** techniques, **machine learning**, and **deep learning** models. It also features a **Flask-based web application** where users can input news content and receive a prediction on whether the news is real or fake.

---

## 🚀 Project Overview

### ✅ Key Features
- **Text Preprocessing:** Lowercasing, punctuation removal, tokenization, stopwords removal, and lemmatization.  
- **Vectorization:** TF-IDF vectorization for converting text data into numerical format.  

### ML Models:
- Logistic Regression  
- Random Forest Classifier  
- Naive Bayes  

### DL Models:
- LSTM (Long Short-Term Memory)  
- CNN (Convolutional Neural Network for Text)  

### Web Application:
- Built using **Flask**  
- Interactive frontend with HTML, CSS, and JavaScript  


---

## 🧠 Machine Learning Workflow

- **Data Loading**  
  Loaded from a structured dataset (e.g., WELFake or Kaggle dataset).

- **Preprocessing**  
  - Lowercasing  
  - Removing punctuation  
  - Tokenization  
  - Stopword removal  
  - Lemmatization

- **TF-IDF Vectorization**  
  Captures the importance of each word in the document relative to the corpus.

- **Model Training & Evaluation**  
  Trained multiple models using training data, evaluated with accuracy, precision, recall, and F1-score.

---

## 🤖 Deep Learning Models

### 1. LSTM
- Used to capture the sequential patterns in text.  
- Word embeddings as input, padded sequences.  
- Trained using binary cross-entropy loss.

### 2. CNN for Text
- Convolutional layers capture n-gram patterns.  
- Max-pooling and dense layers for classification.

---

## 🌐 Web App

The frontend allows users to:  
- Input any news headline or article.  
- Submit the content.  
- View prediction results instantly (Real or Fake).

Flask handles:  
- Backend prediction using pre-trained models.  
- Rendering templates with results.

  ---

## 📁 Webapp Structure

<img width="612" alt="image" src="https://github.com/user-attachments/assets/29a4025d-257e-4b11-9680-17263b37babc" />


## WebApp Preview:
<iframe width="560" height="315" src="https://youtu.be/_lO1Ft5e2_4" frameborder="0" allowfullscreen></iframe>


---

## 💻 How to Run the Project

### 1. Clone the Repository
bash
git clone https://github.com/your-username/FakeNewsDetection.git
cd FakeNewsDetection

### 3. Install Requirements
pip install -r requirements.txt


### 2. Create a Virtual Environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 4. Run the App
python app.py

## 📌 Future Enhancements

- Integrate real-time scraping from news sites.  
- Add BERT or other Transformer-based models.  
- Dockerize the application for easier deployment.  
- Add model selection in the UI.

---

## 📚 Research Connection

This project is inspired by real-world challenges in misinformation and aligns with research efforts in **Natural Language Understanding**, **Fake News Detection**, and **Trustworthy AI**. It showcases how traditional ML models can still perform competitively when well-tuned, and how deep learning models can further enhance contextual understanding in NLP.
