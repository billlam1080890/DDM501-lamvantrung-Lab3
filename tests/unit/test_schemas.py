"""
Unit tests for Pydantic schemas.

Run tests:
    pytest tests/unit/test_schemas.py -v
"""

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
)


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    # =========================================================================
    # Happy path
    # =========================================================================

    def test_valid_request(self):
        """A well-formed request validates and keeps its values."""
        request = PredictionRequest(user_id="196", movie_id="242")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_valid_request_with_numeric_strings(self):
        """IDs are strings; numeric-looking values stay strings."""
        request = PredictionRequest(user_id="1", movie_id="1")
        assert isinstance(request.user_id, str)
        assert isinstance(request.movie_id, str)

    # =========================================================================
    # Missing Field Tests
    # =========================================================================

    def test_missing_user_id_raises_error(self):
        """user_id is required."""
        with pytest.raises(ValidationError):
            PredictionRequest(movie_id="242")

    def test_missing_movie_id_raises_error(self):
        """movie_id is required."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196")

    def test_missing_both_fields_raises_error(self):
        """Both fields missing produces two validation errors, not one."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest()
        assert len(exc_info.value.errors()) == 2

    # =========================================================================
    # Empty / Invalid Input Tests
    # =========================================================================

    def test_empty_user_id_raises_error(self):
        """An empty user_id violates min_length."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="", movie_id="242")

    def test_empty_movie_id_raises_error(self):
        """An empty movie_id violates min_length."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196", movie_id="")

    def test_whitespace_only_user_id_raises_error(self):
        """Whitespace-only ids are rejected by the custom validator."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            PredictionRequest(user_id="   ", movie_id="242")

    def test_none_values_raise_error(self):
        """None is not an acceptable id."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=None, movie_id="242")

    def test_ids_are_stripped(self):
        """Surrounding whitespace is trimmed so ' 196' and '196' are one key."""
        request = PredictionRequest(user_id="  196  ", movie_id=" 242 ")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_too_long_user_id_raises_error(self):
        """max_length caps the id at 50 characters."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="x" * 51, movie_id="242")

    # =========================================================================
    # Type Validation Tests
    # =========================================================================

    def test_integer_user_id_rejected(self):
        """Pydantic v2 is strict about str fields - an int is not coerced."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=196, movie_id="242")


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self):
        """A well-formed response validates."""
        response = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=3.5, model_version="1.0.0"
        )
        assert response.predicted_rating == 3.5

    def test_rating_below_minimum_raises_error(self):
        """A rating under 1.0 is rejected - the contract promises 1-5."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=0.5, model_version="1.0.0"
            )

    def test_rating_above_maximum_raises_error(self):
        """A rating over 5.0 is rejected."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=5.5, model_version="1.0.0"
            )

    def test_rating_at_boundaries(self):
        """Exactly 1.0 and exactly 5.0 are valid - the bounds are inclusive."""
        low = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=1.0, model_version="1.0.0"
        )
        high = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=5.0, model_version="1.0.0"
        )
        assert low.predicted_rating == 1.0
        assert high.predicted_rating == 5.0


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_valid_health_response(self):
        """Health response carries a status string and a load flag."""
        response = HealthResponse(status="healthy", model_loaded=True)
        assert response.status == "healthy"
        assert response.model_loaded is True

    def test_model_loaded_must_be_bool(self):
        """A non-boolean model_loaded is rejected."""
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy", model_loaded="yes please")


class TestBatchSchemas:
    """Tests for the batch request/response schemas."""

    def test_prediction_item_valid(self):
        """A batch item validates like a single request."""
        item = PredictionItem(user_id="196", movie_id="242")
        assert item.user_id == "196"

    def test_valid_batch_request(self):
        """A batch of two items validates."""
        request = BatchPredictionRequest(
            predictions=[
                {"user_id": "196", "movie_id": "242"},
                {"user_id": "186", "movie_id": "302"},
            ]
        )
        assert len(request.predictions) == 2

    def test_empty_batch_raises_error(self):
        """An empty batch is rejected by min_length=1."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=[])

    def test_oversized_batch_raises_error(self):
        """More than 100 items is rejected - this bounds worst-case latency."""
        items = [{"user_id": str(i), "movie_id": "1"} for i in range(101)]
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=items)

    def test_batch_response_total_count(self):
        """The response reports how many predictions it carries."""
        response = BatchPredictionResponse(
            predictions=[
                PredictionResponse(
                    user_id="196", movie_id="242", predicted_rating=3.5, model_version="1.0.0"
                )
            ],
            total_count=1,
        )
        assert response.total_count == 1


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_error_response_default_code(self):
        """error_code defaults so callers always get a machine-readable field."""
        response = ErrorResponse(detail="something broke")
        assert response.error_code == "UNKNOWN_ERROR"

    def test_error_response_custom_code(self):
        """A specific error code overrides the default."""
        response = ErrorResponse(detail="bad input", error_code="VALIDATION_ERROR")
        assert response.error_code == "VALIDATION_ERROR"
