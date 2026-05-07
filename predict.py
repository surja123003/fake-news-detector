import joblib
from preprocess import clean_text

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_news(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]

# User input
news = input("Enter news: ")
print("Prediction:", predict_news(news))