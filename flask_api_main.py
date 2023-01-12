import numpy as np
from flask import Flask, request, jsonify, render_template
import sklearn
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("models/random_forest2_SMOTE.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction == 0:
        return render_template("index.html", prediction_text="The Company will not bankrupt  it's class is {}".format(prediction))
    else:
        return render_template("index.html", prediction_text="The Company will bankrupt  it's class is {}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)