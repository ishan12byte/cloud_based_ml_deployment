from flask import Flask, render_template, request, jsonify
import pickle
import logging

tokenizer=pickle.load(open("models/cv.pkl", "rb"))
model=pickle.load(open("models/clf.pkl","rb"))

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    app.logger.info("Homepage accessed")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("email-content")
    app.logger.info(f"Web prediction request recieved {email_text}")
    tokenized_email=tokenizer.transform([email_text])
    predictions = model.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    app.logger.info(f"Prediction result: {predictions}")
    return render_template("index.html", predictions=predictions, email=email_text)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    email_text = data["email-content"]
    app.logger.info(f"API prediction request recieved {email_text}")
    tokenized_email=tokenizer.transform([email_text])
    predictions = model.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    app.logger.info(f"API prediction result: {predictions}")
    return jsonify({'predictions': predictions, 'email':email_text})

if __name__ == "__main__":
    app.logger.info("Flask application started")
    app.run(host="0.0.0.0", port=8080, debug=True)