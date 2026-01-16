import os
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn.functional as F

# Third-party imports
from flask import Flask, jsonify, render_template, request
from huggingface_hub import snapshot_download
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Load .env if present (optional)
try:
    from dotenv import load_dotenv

    env_path = Path(".") / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    # python-dotenv is optional in some environments; ignore if not installed.
    pass

app = Flask(__name__)

# --------------------------
# Ensure model is available (download from HF Hub if needed)
# --------------------------
MODEL_DIR = os.environ.get("MODEL_PATH", "roberta_chatbot_model")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not os.path.isdir(MODEL_DIR):
    if HF_MODEL_ID:
        try:
            print(f"Model not found locally — downloading {HF_MODEL_ID}...")
            snapshot_download(
                repo_id=HF_MODEL_ID, local_dir=MODEL_DIR, token=HF_TOKEN
            )
        except Exception as e:
            print("Warning: failed to download model from HF Hub:", e)
    else:
        print(
            "Model directory not found and HF_MODEL_ID not set. Expect runtime errors"
        )

model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
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
