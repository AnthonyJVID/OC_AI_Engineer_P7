import pytest
from sentiment_api.utils import clean_text

def test_clean_text():
    # Texte initial comportant différents éléments à nettoyer
    input_text = "Votre texte à nettoyer avec URL, @mentions, #hashtags et emojis 😊"
    cleaned = clean_text(input_text)

    # Vérifier que le texte nettoyé est en minuscules
    assert cleaned == cleaned.lower()

    # Vérifier que certains éléments indésirables ont été supprimés
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned