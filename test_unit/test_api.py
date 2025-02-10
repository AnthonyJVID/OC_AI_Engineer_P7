import pytest
from fastapi.testclient import TestClient
from api.main import app  # Import depuis le module `api`

client = TestClient(app)

def test_predict_endpoint():
    """
    Test de l'endpoint /predict de l'API.
    """
    payload = {"text": "Ceci est un test"}
    response = client.post("/predict", json=payload)

    # Vérification du code de réponse
    assert response.status_code == 200

    data = response.json()

    # Vérification des clés attendues dans la réponse
    assert "text" in data
    assert "probability_positive" in data
    assert "sentiment" in data

    # Vérification de la probabilité
    probability = data["probability_positive"]
    assert 0 <= probability <= 1

    # Vérification de la correspondance entre probabilité et sentiment
    expected_sentiment = "positif" if probability >= 0.5 else "negatif"
    assert data["sentiment"] == expected_sentiment

def test_feedback_endpoint():
    """
    Test de l'endpoint /feedback de l'API.
    """
    # Envoyer un feedback correct
    payload = {"correct": True}
    response = client.post("/feedback", json=payload)

    # Vérifier le code de réponse HTTP
    assert response.status_code == 200

    # Vérifier le contenu de la réponse
    data = response.json()
    assert "message" in data
    assert data["message"] == "Feedback enregistré avec succès."

    # Tester avec un feedback incorrect
    payload = {"correct": False}
    response = client.post("/feedback", json=payload)

    # Vérifier à nouveau le code de réponse HTTP
    assert response.status_code == 200

    # Vérifier le contenu de la réponse
    data = response.json()
    assert "message" in data
    assert data["message"] == "Feedback enregistré avec succès."