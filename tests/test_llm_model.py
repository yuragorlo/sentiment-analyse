from app.models.llm_model import LLMService
from dotenv import load_dotenv

load_dotenv()


class TestLLMService:
    """Tests for the LLMService with real API calls."""
    
    def test_init_service(self):
        """Test LLM service initialization."""
        service = LLMService()
        assert service.openai_client is not None
    
    def test_analyze_with_openai_positive(self):
        """Test analyzing positive text with OpenAI."""
        llm_service = LLMService()
        result = llm_service.analyze_with_openai("I love this product! It's amazing and works perfectly.")
        assert "sentiment" in result
        assert "confidence" in result
        assert result["sentiment"] == "positive"
        assert result["confidence"] >= 0.7
    
    def test_analyze_with_openai_negative(self):
        """Test analyzing negative text with OpenAI."""
        llm_service = LLMService()
        result = llm_service.analyze_with_openai("I hate this product. It's terrible and doesn't work at all.")
        assert "sentiment" in result
        assert "confidence" in result
        assert result["sentiment"] == "negative"
        assert result["confidence"] >= 0.7
    
    def test_analyze_with_openai_neutral(self):
        """Test analyzing neutral text with OpenAI."""
        llm_service = LLMService()
        result = llm_service.analyze_with_openai("This product is average. It works as expected, nothing special.")
        assert "sentiment" in result
        assert "confidence" in result
        assert result["sentiment"] == "neutral"
    
    def test_analyze_complex_text(self):
        """Test analyzing complex text with mixed sentiments."""
        llm_service = LLMService()
        complex_text = """
        I have mixed feelings about this product. The design is excellent and the materials are high quality.
        However, the customer service was disappointing and delivery took too long.
        Overall, it's decent value for money but could be improved.
        """
        result = llm_service.analyze_with_openai(complex_text)
        assert "sentiment" in result
        assert "confidence" in result