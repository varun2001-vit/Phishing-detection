import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))
def preprocess_text(text):
    # Tokenize the text
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    data = tokenizer.tokenize(text)
    # Stem the words
    stemmer = SnowballStemmer("english")
    stemmed_data = [stemmer.stem(word) for word in data]
    # Join the stemmed words
    joined_data = ' '.join(stemmed_data)
    # Vectorize the data
    vectorized_data = cv.transform([joined_data])
    return vectorized_data
@app.route('/')   
def home():
   return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    final_features = preprocess_text(url)
    prediction = model.predict(final_features)
    if prediction[0] == 0:
        message = "URL is vulnerable and prone to attack"
    else:
        message = "URL is not vulnerable to  attack"
    return render_template('index.html', prediction_text=message)

if __name__ == '__main__':
   app.run(debug=True)