# Enhanced Text Analysis API

A Dockerized FastAPI service for text analysis using both traditional machine learning and modern AI language models (LLMs).

## Features

- Traditional ML sentiment analysis using scikit-learn
- LLM-based text analysis with advanced capabilities
- REST API with JSON responses
- Request validation using Pydantic
- Caching mechanism for efficient response handling
- Container-based deployment with Docker
- Comprehensive testing suite

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9 or higher (for local development)
- API keys for LLM providers (OpenAI and/or Anthropic)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yuragorlo/sentiment-analyse
   cd enhanced-text-analysis-api
   ```

2. Create a `.env` file by copying the example:
   ```bash
   cp .env.example .env
   ```

3. Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running with Docker

Build and start the service:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python -m app.main
   ```

## API Endpoints

### Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Main Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check
- `POST /api/predict`: Predict sentiment using the traditional ML model
- `POST /api/predict-llm`: Predict sentiment using the LLM

## Testing

Run the test suite:

```bash
pytest
```

### Manual Testing

Test the ML model endpoint:
```bash
./tests/positive_test.sh
./tests/negative_test.sh
```

Test the LLM model endpoint:
```bash
./tests/llm_positive_test.sh
./tests/llm_negative_test.sh
```

## Project Structure

```
.
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── routes.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_model.py
│   │   └── ml_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   └── config.py
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── llm_negative_test.sh
│   ├── llm_positive_test.sh
│   ├── negative_test.sh
│   ├── positive_test.sh
│   ├── test_api.py
│   ├── test_llm_model.py
│   └── test_ml_model.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```