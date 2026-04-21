import os
import pickle as pkl
from flask import Flask, jsonify, request
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("model") / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found {MODEL_PATH}")
with open(MODEL_PATH,"rb") as f:
    model = pkl.load(f)


@app.route("/",methods=["GET"])
def home():
    return jsonify({
        "message":"Python AI model running on sap BTP Cloud foundry"
    })   

@app.route("/health",methods=["GET"])
def health():
    return jsonify({
        "status":"UP"
    })

user_string = "Hello, World!"

for i in user_string:
    print(i)    
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        features = payload.get("features")
        if not isinstance(features,list) or len(features)!=4:
            return jsonify({
                "error":"features must be a list value of length 4"
            }),400
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0].tolist()

        return jsonify({
            "prediction":int(prediction),
            "probability":probabilities
        })
    except Exception as ex:
        return jsonify({
            "error":str(ex)
        }),500
    
if __name__ == "__main__":

    port = int(os.getenv("PORT", 8080))

    app.run(host="0.0.0.0", port=port)
