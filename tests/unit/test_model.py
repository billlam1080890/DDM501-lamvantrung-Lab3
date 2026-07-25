"""
Unit tests for MovieRatingModel class.

Run tests:
    pytest tests/unit/test_model.py -v
"""

import pytest

from app.config import MAX_RATING, MIN_RATING
from app.model import MovieRatingModel


class TestMovieRatingModel:
    """Unit tests for MovieRatingModel class."""

    # =========================================================================
    # Model Loading Tests
    # =========================================================================

    def test_model_loads_successfully(self, trained_model):
        """Test that model loads without errors."""
        assert trained_model is not None
        assert trained_model.is_loaded()

    def test_model_instance_has_model_attribute(self, trained_model):
        """Test that model instance has the model attribute."""
        assert hasattr(trained_model, "model")
        assert trained_model.model is not None

    # =========================================================================
    # Prediction Return Type Tests
    # =========================================================================

    def test_predict_returns_float(self, trained_model):
        """predict() returns a float, not a Surprise Prediction object."""
        result = trained_model.predict("196", "242")
        assert isinstance(result, float)

    # =========================================================================
    # Rating Range Tests
    # =========================================================================

    def test_predict_returns_value_in_valid_range(self, trained_model):
        """A single prediction is clipped into the 1-5 rating scale."""
        result = trained_model.predict("196", "242")
        assert MIN_RATING <= result <= MAX_RATING

    def test_predict_multiple_pairs_all_in_range(self, trained_model, known_user_movie_pairs):
        """Every known pair predicts inside the rating scale."""
        for pair in known_user_movie_pairs:
            result = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert (
                MIN_RATING <= result <= MAX_RATING
            ), f"Out of range for user={pair['user_id']} movie={pair['movie_id']}: {result}"

    # =========================================================================
    # Batch Prediction Tests
    # =========================================================================

    def test_predict_batch_returns_list(self, trained_model):
        """predict_batch() returns a list."""
        pairs = [("196", "242"), ("186", "302")]
        result = trained_model.predict_batch(pairs)
        assert isinstance(result, list)

    def test_predict_batch_returns_correct_length(self, trained_model):
        """One prediction is returned per input pair."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        result = trained_model.predict_batch(pairs)
        assert len(result) == len(pairs)

    def test_predict_batch_all_values_in_range(self, trained_model):
        """Batch predictions respect the same clipping as single predictions."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        for rating in trained_model.predict_batch(pairs):
            assert MIN_RATING <= rating <= MAX_RATING

    def test_predict_batch_empty_list_returns_empty(self, trained_model):
        """An empty batch is valid and returns an empty list."""
        assert trained_model.predict_batch([]) == []

    # =========================================================================
    # is_loaded() Tests
    # =========================================================================

    def test_is_loaded_returns_bool(self, trained_model):
        """is_loaded() returns an actual bool, not a truthy object."""
        assert isinstance(trained_model.is_loaded(), bool)

    def test_is_loaded_returns_true_for_loaded_model(self, trained_model):
        """A successfully constructed model reports itself as loaded."""
        assert trained_model.is_loaded() is True

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_predict_with_none_user_id(self, trained_model):
        """A None id must not crash: Surprise treats it as an unknown user."""
        result = trained_model.predict(None, "242")
        assert MIN_RATING <= result <= MAX_RATING

    def test_predict_with_empty_string(self, trained_model):
        """An empty id falls back to the global mean rather than raising."""
        result = trained_model.predict("", "")
        assert MIN_RATING <= result <= MAX_RATING

    def test_predict_raises_when_model_not_loaded(self, trained_model):
        """Calling predict() on an unloaded wrapper raises RuntimeError."""
        broken = MovieRatingModel.__new__(MovieRatingModel)
        broken.model = None

        with pytest.raises(RuntimeError, match="Model not loaded"):
            broken.predict("196", "242")


class TestModelFileHandling:
    """Tests for model file loading behaviour."""

    def test_model_raises_error_for_missing_file(self):
        """A missing model file fails fast at construction time."""
        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path="models/does_not_exist.pkl")

    def test_model_raises_for_corrupt_file(self, tmp_path):
        """A file that is not a valid pickle is rejected, not silently ignored."""
        corrupt = tmp_path / "corrupt.pkl"
        corrupt.write_text("this is not a pickle")

        with pytest.raises(Exception):
            MovieRatingModel(model_path=str(corrupt))


class TestModelSingleton:
    """Tests for the module-level model singleton."""

    def test_get_model_returns_same_instance(self):
        """get_model() caches - loading a 5 MB pickle per request would be wasteful."""
        from app.model import get_model, reset_model

        reset_model()
        first = get_model()
        second = get_model()

        assert first is second

    def test_reset_model_clears_instance(self):
        """reset_model() forces the next get_model() to rebuild."""
        from app.model import get_model, reset_model

        reset_model()
        first = get_model()
        reset_model()
        second = get_model()

        assert first is not second
