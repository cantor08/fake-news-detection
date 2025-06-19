import re
from flask import Flask, request, render_template
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import joblib
from scipy.sparse import hstack

# Ensure stopwords are downloaded
nltk.download('stopwords', quiet=True)

# ========== Text Preprocessing ==========
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# ========== Load Artifacts ==========
try:
    vectorizers = joblib.load("vectorizer.pkl")
    vectorizer_title = vectorizers['title']
    vectorizer_text = vectorizers['text']
    model = joblib.load("fake_news_logreg_model.pkl")
    print("Model and vectorizers loaded successfully")
    print(f"Model classes: {model.classes_}")
except Exception as e:
    print(f"Error loading artifacts: {str(e)}")
    exit()

# ========== Flask App ==========
app = Flask(__name__)

def debug_prediction(title_vec, text_vec, combined_features):
    """Debug helper function"""
    print("\n=== DEBUG INFO ===")
    print(f"Title non-zero features: {title_vec.nnz}")
    print(f"Text non-zero features: {text_vec.nnz}")
    print(f"Combined shape: {combined_features.shape}")
    print(f"Feature sum: {combined_features.sum()}")
    if hasattr(model, 'coef_'):
        print(f"Model coefficients shape: {model.coef_.shape}")

@app.route('/')
def home():
    return render_template('index.html', 
                         prediction_text=None, 
                         probability_breakdown=None,
                         original_title="",
                         original_text="")

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title', '')
    description = request.form.get('description', '')
    
    if not title.strip() or not description.strip():
        return render_template('index.html',
                            prediction_text="Please enter both title and content",
                            probability_breakdown=None,
                            original_title=title,
                            original_text=description)
    
    try:
        # Preprocess text
        processed_title = preprocess_text(title)
        processed_text = preprocess_text(description)
        
        print(f"\nProcessed Title: {processed_title}")
        print(f"Processed Text: {processed_text[:100]}...")
        
        # Vectorize inputs
        title_vec = vectorizer_title.transform([processed_title])
        text_vec = vectorizer_text.transform([processed_text])
        combined_features = hstack([title_vec, text_vec])
        
        debug_prediction(title_vec, text_vec, combined_features)
        
        # Make prediction
        prediction = model.predict(combined_features)[0]
        probabilities = model.predict_proba(combined_features)[0]
        
        # Prepare results
        label_map = {0: "FAKE", 1: "REAL"}
        breakdown = {
            label_map[0]: round(probabilities[0] * 100, 2),
            label_map[1]: round(probabilities[1] * 100, 2)
        }

        #result = f"{'🟢 REAL News' if prediction == 1 else '🔴 FAKE News'} (Confidence: {breakdown[label_map[prediction]]}%)"
        result = f"{breakdown[label_map[prediction]]}% confidence"  # Or empty string if you don't want any text
        
        return render_template('index.html', 
            prediction_text=result, 
            probability_breakdown=breakdown,
            original_title=title,
            original_text=description,
            prediction=prediction  # Add this line to pass the prediction value
)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template('index.html',
                            prediction_text=f"Error: {str(e)}",
                            probability_breakdown=None,
                            original_title=title,
                            original_text=description)
                            

if __name__ == '__main__':
    app.run(debug=True, port=5000)