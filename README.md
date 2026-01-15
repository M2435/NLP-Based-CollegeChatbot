# College Chatbot

A Flask-based chatbot application powered by RoBERTa for answering college-related queries.

## Features

- NLP-based intent classification using RoBERTa
- Web-based interface with Flask
- CSV-based intent and response management

# College Chatbot

A production-ready Flask chatbot focused on NLP-based intent classification using RoBERTa. This repository contains the Flask app, training/fine-tuning scripts, and example intent data.

**Status:** Prototype → Ready for deployment after model hosting configuration.

## Key Features

- NLP-based intent classification using RoBERTa
- Lightweight Flask web UI
- CSV-based intent/response store for easy editing
- Dockerfile and CI workflow included

## Project Structure

- `app.py` — Main Flask application and inference endpoint
- `train_roberta_chatbot.py` — Script to fine-tune RoBERTa on `intents.csv` (optional)
- `train_chatbot.py` — Alternate training script
- `intents.csv` — Patterns and tag-based responses
- `templates/` — Web UI templates (HTML)
- `roberta_chatbot_model/` — Local model directory (do NOT commit large artifacts)

## Quick start (local)

1. Create and activate a virtual environment:

```bash
python -m venv chatbotenv
# Windows
chatbotenv\Scripts\activate
# macOS / Linux
source chatbotenv/bin/activate
```

2. Install runtime dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app (development):

```bash
export FLASK_DEBUG=true             # or set in .env on Windows
python app.py
```

Visit: http://localhost:5000

## Docker (recommended for deployment)

Build and run the container locally:

```bash
docker build -t college-chatbot:latest .
docker run -p 5000:5000 college-chatbot:latest
```

The container runs the app under `gunicorn` on port 5000.

## Deploying to Heroku (example)

1. Create the app on Heroku:

```bash
heroku create your-app-name
git push heroku main
```

Heroku will use the `Procfile` and run `gunicorn` automatically.

## Model hosting and large artifacts

- Do not commit `roberta_chatbot_model/` to GitHub. Instead:
  - Use Git LFS for model weights, OR
  - Host the model in cloud storage (S3, GCS) and download at startup, OR
  - Use a model registry (Hugging Face Hub) and download at container startup.

- If the model is stored externally, update `app.py` to load from a path or URL and include authentication via environment variables.

## Environment variables

- `HOST` — host to bind (default `0.0.0.0`)
- `PORT` — port (default `5000`)
- `FLASK_DEBUG` — set to `true` for debug mode (not for production)

Create a `.env` (not tracked) or use your platform's secrets manager for production.

## Fine-tuning the model (optional)

To fine-tune RoBERTa locally on the provided `intents.csv` dataset:

```bash
python train_roberta_chatbot.py
```

This will save the fine-tuned model to `roberta_chatbot_model/` and `label_encoder.pkl`.

## Contributing

- Fork the repo, create a feature branch, and open a pull request.
- Please run `black .`, `isort .`, and `flake8` locally before opening PRs.

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
