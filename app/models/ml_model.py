import os
import pickle
import pandas as pd
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from datasets import load_dataset
from app.utils.config import get_model_path
from app.utils.logger import get_logger

logger = get_logger()

class MLModel:
    """Sentiment analysis model using TF-IDF and Logistic Regression."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or get_model_path()
        self.vectorizer = None
        self.model = None
        self.loaded = False

    def load(self) -> bool:
        if self.loaded:
            return True

        model_path = Path(self.model_path)
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.model = model_data['model']
                    self.loaded = True
                    return True
            except (pickle.PickleError, IOError, KeyError) as e:
                logger.error(f"Error loading model: {e}")
        self._train_new_model()
        return self.loaded

    def _train_new_model(self) -> None:
        """Train a new sentiment analysis model using a dataset from Hugging Face."""
        try:
            dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")
            df = pd.DataFrame(dataset['train'])
            texts = df['text'].tolist()
            labels = df['sentiment'].tolist()

            self.vectorizer = TfidfVectorizer(max_features=5000)
            X = self.vectorizer.fit_transform(texts)
            self.model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
            self.model.fit(X, labels)

            model_data = {'vectorizer': self.vectorizer, 'model': self.model}
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            try:
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                self.loaded = True
            except IOError as e:
                logger.error(f"Error saving model: {e}")
                self.loaded = True
        except Exception as e:
            logger.error(f"Error training model with Hugging Face dataset: {e}")

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment from text."""
        if not self.loaded:
            success = self.load()
            if not success:
                raise RuntimeError("Failed to load sentiment model")

        X = self.vectorizer.transform([text])
        sentiment = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        confidence = max(probs)

        return sentiment, float(confidence)


ml_model = MLModel()
