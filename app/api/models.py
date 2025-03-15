from pydantic import BaseModel, Field,  field_validator


class MLAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., min_length=1, max_length=5000, description="The text to analyze")

    @field_validator('text')
    def text_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Text must not be empty')
        return v
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "text": "I love this product! It's amazing and works perfectly."
            }
        }

class MLAnalysisResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str = Field(..., description="The predicted sentiment (positive, negative, neutral)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the prediction")
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.95
            }
        }

class LLMAnalysisRequest(BaseModel):
    """Request model for LLM-based text analysis."""
    text: str = Field(..., min_length=1, max_length=5000, description="The text to analyze")
    
    @field_validator('text')
    def text_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Text must not be empty')
        return v
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "text": "I love this product! It's amazing and works perfectly.",
            }
        }

class LLMAnalysisResponse(BaseModel):
    """Response model for LLM-based text analysis."""
    sentiment: str = Field(..., description="The predicted sentiment (positive, negative, neutral)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the prediction")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.92,
            }
        }

class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error details")
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "detail": "An error occurred while processing the request."
            }
        } 