import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Load .env if present
try:
    from dotenv import load_dotenv

    env_path = Path(".") / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    # python-dotenv is optional in some environments.
    # Ignore if not installed.
    pass

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer

app = Flask(__name__)

# --------------------------
# Load trained RoBERTa model & tokenizer
# --------------------------
model = RobertaForSequenceClassification.from_pretrained("roberta_chatbot_model")
tokenizer = RobertaTokenizer.from_pretrained("roberta_chatbot_model")
encoder = joblib.load("label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load intents for response lookup
data = pd.read_csv("intents.csv")


# --------------------------
# Flask routes
# --------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def get_bot_response(user_input):
    # Tokenize user input
    encoding = tokenizer.encode_plus(
        user_input,
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        intent = encoder.inverse_transform([pred_idx])[0]

    # Fetch response from intents.csv
    responses = data[data["tag"] == intent]["response"].tolist()
    if responses:
        return responses[0]  # You can randomize if multiple responses
    else:
        return "I'm not sure I understand. Can you rephrase?"


@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.json.get("message", "")
    reply = get_bot_response(user_input)
    return jsonify({"response": reply})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
