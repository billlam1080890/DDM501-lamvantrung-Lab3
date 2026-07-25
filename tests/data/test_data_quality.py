"""
Data quality validation tests.

These validate the *data* rather than the code. In an ML system bad data is
as damaging as a bug, and it fails silently - a model trains happily on
corrupted ratings and simply gets worse.

Run tests:
    pytest tests/data/test_data_quality.py -v
"""

import numpy as np

MIN_VALID_RATING = 1.0
MAX_VALID_RATING = 5.0
REQUIRED_FIELDS = ("user_id", "movie_id", "rating")


class TestRatingDataQuality:
    """Schema and range checks on rating records."""

    # =========================================================================
    # Rating Range Tests
    # =========================================================================

    def test_all_ratings_in_valid_range(self, sample_ratings):
        """Every rating sits inside the 1-5 scale."""
        for record in sample_ratings:
            assert (
                MIN_VALID_RATING <= record["rating"] <= MAX_VALID_RATING
            ), f"rating {record['rating']} out of range in {record}"

    def test_no_negative_ratings(self, sample_ratings):
        """No rating is negative."""
        assert all(record["rating"] >= 0 for record in sample_ratings)

    def test_no_ratings_above_maximum(self, sample_ratings):
        """No rating exceeds the maximum of the scale."""
        assert max(r["rating"] for r in sample_ratings) <= MAX_VALID_RATING

    def test_ratings_are_numeric(self, sample_ratings):
        """Ratings are numbers - a string '4.0' would break aggregation."""
        for record in sample_ratings:
            assert isinstance(record["rating"], (int, float))
            assert not isinstance(record["rating"], bool)

    # =========================================================================
    # ID Validation Tests
    # =========================================================================

    def test_no_missing_user_ids(self, sample_ratings):
        """Every record has a non-empty user_id."""
        for record in sample_ratings:
            assert record.get("user_id"), f"missing user_id in {record}"

    def test_no_missing_movie_ids(self, sample_ratings):
        """Every record has a non-empty movie_id."""
        for record in sample_ratings:
            assert record.get("movie_id"), f"missing movie_id in {record}"

    def test_user_ids_are_strings(self, sample_ratings):
        """IDs are strings, matching what the model was trained on."""
        assert all(isinstance(r["user_id"], str) for r in sample_ratings)

    def test_movie_ids_are_strings(self, sample_ratings):
        """Movie IDs are strings for the same reason."""
        assert all(isinstance(r["movie_id"], str) for r in sample_ratings)

    def test_ids_have_no_surrounding_whitespace(self, sample_ratings):
        """' 196' and '196' must not become two different users."""
        for record in sample_ratings:
            assert record["user_id"] == record["user_id"].strip()
            assert record["movie_id"] == record["movie_id"].strip()

    # =========================================================================
    # Data Completeness Tests
    # =========================================================================

    def test_no_null_ratings(self, sample_ratings):
        """No rating is null."""
        assert all(record["rating"] is not None for record in sample_ratings)

    def test_all_records_have_required_fields(self, sample_ratings):
        """Every record carries the full schema."""
        for record in sample_ratings:
            for field in REQUIRED_FIELDS:
                assert field in record, f"'{field}' missing from {record}"

    def test_dataset_is_not_empty(self, sample_ratings):
        """An empty dataset would make every downstream test vacuously pass."""
        assert len(sample_ratings) > 0

    def test_no_duplicate_user_movie_pairs(self, sample_ratings):
        """A user rating the same movie twice is a data integrity error."""
        pairs = [(r["user_id"], r["movie_id"]) for r in sample_ratings]
        assert len(pairs) == len(set(pairs)), "duplicate (user, movie) pairs found"


class TestRatingDistribution:
    """Statistical checks - these catch corruption that schema checks miss."""

    # =========================================================================
    # Distribution Tests
    # =========================================================================

    def test_mean_rating_reasonable(self, sample_ratings):
        """The mean sits inside the scale, not at a degenerate edge."""
        mean = np.mean([r["rating"] for r in sample_ratings])
        assert MIN_VALID_RATING <= mean <= MAX_VALID_RATING

    def test_rating_standard_deviation(self, sample_ratings):
        """Some spread exists.

        Zero variance means every rating is identical, which signals a broken
        export rather than real user behaviour.
        """
        stdev = np.std([r["rating"] for r in sample_ratings])
        assert stdev > 0.0

    def test_multiple_rating_values_exist(self, sample_ratings):
        """More than one distinct rating value is present."""
        assert len({r["rating"] for r in sample_ratings}) > 1

    def test_multiple_users_present(self, sample_ratings):
        """Collaborative filtering needs more than one user to learn anything."""
        assert len({r["user_id"] for r in sample_ratings}) > 1

    def test_multiple_movies_present(self, sample_ratings):
        """Likewise more than one item."""
        assert len({r["movie_id"] for r in sample_ratings}) > 1


class TestPredictionOutputQuality:
    """Data quality applied to model output, not just model input."""

    def test_predictions_are_in_scale(self, trained_model, known_user_movie_pairs):
        """Served predictions obey the same range rule as training data."""
        for pair in known_user_movie_pairs:
            rating = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert MIN_VALID_RATING <= rating <= MAX_VALID_RATING

    def test_predictions_have_no_nan(self, trained_model, known_user_movie_pairs):
        """A NaN prediction would serialise to invalid JSON and break clients."""
        for pair in known_user_movie_pairs:
            rating = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert not np.isnan(rating), "prediction is NaN"

    def test_prediction_distribution_not_degenerate(self, trained_model, known_user_movie_pairs):
        """The model does not collapse to a single constant output."""
        ratings = [
            trained_model.predict(p["user_id"], p["movie_id"]) for p in known_user_movie_pairs
        ]
        assert len(set(ratings)) > 1, f"all predictions identical: {ratings[0]}"
