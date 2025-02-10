import pytest
from api.main import predict_sentiment  # Import depuis le module `api`

def test_predict_sentiment_function():
    """
    Test de la fonction predict_sentiment en isolation.
    """
    text = "Ceci est un test"
    probability, sentiment = predict_sentiment(text)

    # Vérification de la probabilité
    assert 0 <= probability <= 1

    # Vérification de la cohérence entre la probabilité et le sentiment
    expected_sentiment = "positif" if probability >= 0.5 else "negatif"
    assert sentiment == expected_sentiment