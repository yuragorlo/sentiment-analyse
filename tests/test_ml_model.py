from app.models.ml_model import MLModel

def test_sentiment_model_prediction():
    """Test sentiment model prediction."""
    model = MLModel()

    positive_text = "I love this product! It's amazing."
    sentiment, confidence = model.predict(positive_text)
    assert sentiment == "positive"
    assert 0.5 <= confidence <= 1.0

    negative_text = "This is terrible. I hate it."
    sentiment, confidence = model.predict(negative_text)
    assert sentiment == "negative"
    assert 0.5 <= confidence <= 1.0

    neutral_text = "This is okay I guess. Nothing special."
    sentiment, confidence = model.predict(neutral_text)
    assert sentiment == "neutral"
    assert 0.0 <= confidence <= 1.0

def test_model_loading():
    model = MLModel()
    assert model.load() == True
    assert model.loaded == True 