import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datasets import load_dataset
from app.utils.logger import get_logger


logger = get_logger()

def tune_sentiment_classifier():
    """
    Tune hyperparameters for the sentiment analysis classifier using GridSearchCV.
    The function splits data into train/test sets, defines parameter grid,
    performs grid search, and reports the best parameters and performance.
    """
    try:
        dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")
        df = pd.DataFrame(dataset['train'])

        texts = df['text'].tolist()
        labels = df['sentiment'].tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression(max_iter=3000))
        ])

        param_grid = {
            'tfidf__max_features': [3000, 5000, 10000],
            'tfidf__min_df': [1, 2, 5],
            'tfidf__max_df': [0.9, 0.95, 0.99],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 0.5, 1.0, 5.0, 10.0],
            'classifier__class_weight': ['balanced', None],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__penalty': ['l1', 'l2']
        }
        
        logger.info("Starting grid search. This may take some time...")
        grid_search = RandomizedSearchCV(
            pipeline, 
            param_grid, 
            n_iter=30,
            cv=5,
            scoring='f1_macro',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        logger.info(f"Best parameters: {best_params}")

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        logger.info(f"Test set accuracy: {accuracy:.4f}")
        logger.info(f"Test set F1-macro: {f1:.4f}")
        logger.info("\nDetailed classification report:")
        logger.info(classification_report(y_test, y_pred))

        return {
            'best_parameters': best_params,
            'test_accuracy': accuracy,
            'test_f1_macro': f1
        }
        
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        raise


if __name__ == "__main__":
    results = tune_sentiment_classifier()
    print("\n" + "="*50)
    print("BEST HYPERPARAMETERS:")
    for param, value in results['best_parameters'].items():
        print(f"{param}: {value}")
    print("\nPERFORMANCE:")
    print(f"Accuracy: {results['test_accuracy']:.4f}")
    print(f"F1-macro: {results['test_f1_macro']:.4f}")
    print("="*50) 