"""from flask import Flask, request, render_template
import joblib
from preprocess import clean_text

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)[0]
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)"""

from flask import Flask, request, render_template
import joblib
from preprocess import clean_text

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Get user news
    news = request.form['news']

    # Clean text
    cleaned = clean_text(news)

    # Convert to vector
    vector = vectorizer.transform([cleaned])

    # Predict
    result = model.predict(vector)[0]

    # Better output
    if result == "FAKE":
        prediction = "❌ This news appears FAKE"
    else:
        prediction = "✅ This news appears REAL"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)