import os
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace

# Suppression des messages relatifs à TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Définition des chemins des fichiers
BASE_DIR = os.path.dirname(__file__)  # Répertoire contenant le fichier main.py
MODEL_PATH = os.path.join(BASE_DIR, "../models/Glove_gru_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "../models/tokenizer.pickle")
MAXLEN_PATH = os.path.join(BASE_DIR, "../models/maxlen.pickle")

# Vérifier l'existence des fichiers
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer introuvable : {TOKENIZER_PATH}")
if not os.path.exists(MAXLEN_PATH):
    raise FileNotFoundError(f"Maxlen introuvable : {MAXLEN_PATH}")

# Charger le modèle, le tokenizer et la longueur maximale
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

with open(MAXLEN_PATH, "rb") as handle:
    max_len = pickle.load(handle)

# 1) Définition des schémas Pydantic
class TweetRequest(BaseModel):
    text: str

class UserFeedbackRequest(BaseModel):
    correct: bool

# 2) Instancier l'application FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API opérationnelle"}

# 3) Configuration Azure Monitor
tracer_provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="https://uksouth.monitor.azure.com")
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# 4) Fonction de prédiction
def predict_sentiment(text: str):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prob = model.predict(pad)[0][0]
    sentiment_label = "positif" if prob >= 0.5 else "negatif"
    return float(prob), sentiment_label

# 5) Endpoint POST -> /predict
@app.post("/predict")
def get_prediction(tweet_req: TweetRequest):
    text = tweet_req.text
    probability, label = predict_sentiment(text)

    # Enregistrer la prédiction dans Azure
    with tracer.start_as_current_span("prediction_sentiment") as span:
        span.set_attribute("text", text)
        span.set_attribute("probability_positive", probability)
        span.set_attribute("predicted_sentiment", label)

    return {
        "text": text,
        "probability_positive": probability,
        "sentiment": label
    }

# 6) Endpoint POST -> /feedback
@app.post("/feedback")
def user_feedback(feedback: UserFeedbackRequest):
    with tracer.start_as_current_span("user_feedback") as span:
        span.set_attribute("user_validation", feedback.correct)

    return {"message": "Feedback enregistré avec succès."}

# 7) Application en local (développement)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)