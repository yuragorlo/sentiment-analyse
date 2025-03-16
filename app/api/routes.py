import traceback
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from app.api.models import MLAnalysisRequest, MLAnalysisResponse, LLMAnalysisRequest, LLMAnalysisResponse, ErrorResponse
from app.models.ml_model import ml_model
from app.models.llm_model import llm_model
from app.utils.cache import cache
from app.utils.logger import get_logger

logger = get_logger()

router = APIRouter()

@router.post(
    "/predict", 
    response_model=MLAnalysisResponse,
    responses={
        200: {"model": MLAnalysisResponse, "description": "Successful prediction"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def predict_sentiment(request: MLAnalysisRequest):
    """
    Predict sentiment of the provided text using a traditional ML model.
    """
    try:
        input_text = request.text
        cached_response = cache.get("predict", {"text": input_text})
        if cached_response:
            return JSONResponse(content=cached_response)

        sentiment, confidence = ml_model.predict(input_text)
        response = MLAnalysisResponse(sentiment=sentiment, confidence=confidence)
        response_dict = jsonable_encoder(response)
        cache.set("predict", {"text": input_text}, response_dict)
        logger.info(f"Sentiment for {input_text=} was predicted by ML: {response_dict=}")
        return JSONResponse(content=response_dict)
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Error in predict_sentiment: {traceback_str}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing the request: {str(e)}"
        )

@router.post(
    "/predict-llm", 
    response_model=LLMAnalysisResponse,
    responses={
        200: {"model": LLMAnalysisResponse, "description": "Successful prediction"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def predict_sentiment_llm(request: LLMAnalysisRequest):
    """
    Predict sentiment of the provided text using a Large Language Model.
    """
    try:
        input_text = request.text
        cached_response = cache.get("predict-llm", {"text": input_text})
        if cached_response:
            return JSONResponse(content=cached_response)

        result = llm_model.analyze_with_openai(input_text)
        response = LLMAnalysisResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
        )
        response_dict = jsonable_encoder(response)
        cache.set("predict-llm", {"text": input_text}, response_dict)
        logger.info(f"Sentiment for {input_text=} was predicted by LLM: {response_dict=}")
        return JSONResponse(content=response_dict)
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Error in predict_sentiment_llm: {traceback_str}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing the request: {str(e)}"
        ) 