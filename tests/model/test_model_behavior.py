"""
Model behavioral tests.

These tests verify the model's behavior patterns:
- Invariance: Output shouldn't change for certain perturbations
- Directional: Output should change in expected direction
- Minimum Functionality: Basic cases the model must handle

Accuracy metrics alone cannot catch these failures: a model can hold its RMSE
while returning a different answer for the same input, or collapsing to one
constant value for every user.

Run tests:
    pytest tests/model/test_model_behavior.py -v
"""


class TestModelInvariance:
    """Output must not change under perturbations that carry no information."""

    # =========================================================================
    # Deterministic Output Tests
    # =========================================================================

    def test_same_input_same_output(self, trained_model):
        """Identical inputs give identical outputs.

        SVD inference is pure arithmetic over learned factors; any variation
        would mean state is leaking between calls.
        """
        first = trained_model.predict("196", "242")
        second = trained_model.predict("196", "242")
        assert first == second

    def test_multiple_calls_consistent(self, trained_model):
        """Ten consecutive calls return exactly one distinct value."""
        results = {trained_model.predict("196", "242") for _ in range(10)}
        assert len(results) == 1, f"non-deterministic output: {results}"

    def test_interleaved_predictions_do_not_interfere(self, trained_model):
        """Predicting for another pair in between does not change the result."""
        baseline = trained_model.predict("196", "242")
        trained_model.predict("186", "302")
        assert trained_model.predict("196", "242") == baseline

    # =========================================================================
    # Batch Order Invariance Tests
    # =========================================================================

    def test_batch_order_independent(self, trained_model):
        """Reordering a batch reorders the results, it does not change them."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        forward = trained_model.predict_batch(pairs)
        backward = trained_model.predict_batch(list(reversed(pairs)))

        assert forward == list(reversed(backward))

    def test_individual_vs_batch_same_results(self, trained_model):
        """Batching is an optimisation, not a different code path."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]

        batch = trained_model.predict_batch(pairs)
        individual = [trained_model.predict(user, movie) for user, movie in pairs]

        assert batch == individual

    def test_batch_size_does_not_affect_values(self, trained_model):
        """A pair predicts the same whether alone or inside a large batch."""
        alone = trained_model.predict_batch([("196", "242")])[0]
        in_batch = trained_model.predict_batch(
            [("196", "242"), ("186", "302"), ("22", "377"), ("244", "51")]
        )[0]
        assert alone == in_batch


class TestModelDirectional:
    """Output must vary in the expected direction as inputs vary."""

    # =========================================================================
    # Directional Tests
    # =========================================================================

    def test_predictions_are_reasonable(self, trained_model, known_user_movie_pairs):
        """Predictions cluster in the plausible middle of the scale.

        A model predicting 1.0 or 5.0 everywhere would still 'work' but is
        useless for ranking.
        """
        for pair in known_user_movie_pairs:
            rating = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert 1.0 <= rating <= 5.0
            assert 1.5 <= rating <= 4.8, f"implausible extreme prediction {rating}"

    def test_different_movies_different_predictions(self, trained_model):
        """One user rating different movies does not give one flat value."""
        user = "196"
        ratings = [trained_model.predict(user, movie) for movie in ("242", "302", "377", "51")]
        assert len(set(ratings)) > 1, f"same prediction for every movie: {ratings[0]}"

    def test_different_users_different_predictions(self, trained_model):
        """Different users rating one movie differ - otherwise personalisation
        is not happening and the model is just an item-average lookup."""
        movie = "242"
        ratings = [trained_model.predict(user, movie) for user in ("196", "186", "22", "244")]
        assert len(set(ratings)) > 1, f"same prediction for every user: {ratings[0]}"

    def test_unknown_user_falls_back_to_neutral(self, trained_model):
        """A never-seen user gets a global-mean-ish estimate, not an extreme."""
        rating = trained_model.predict("this_user_does_not_exist", "242")
        assert 2.5 <= rating <= 4.5


class TestMinimumFunctionality:
    """The simplest cases the model must never get wrong."""

    # =========================================================================
    # Minimum Functionality Tests
    # =========================================================================

    def test_can_predict_for_known_user(self, trained_model):
        """A user from the training set predicts successfully."""
        rating = trained_model.predict("196", "242")
        assert rating is not None
        assert 1.0 <= rating <= 5.0

    def test_can_predict_for_multiple_users(self, trained_model, known_user_movie_pairs):
        """Every known pair produces a usable prediction."""
        for pair in known_user_movie_pairs:
            rating = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert isinstance(rating, float)
            assert 1.0 <= rating <= 5.0

    def test_predictions_not_all_same(self, trained_model, known_user_movie_pairs):
        """Across known pairs the model produces varied output."""
        ratings = [
            trained_model.predict(p["user_id"], p["movie_id"]) for p in known_user_movie_pairs
        ]
        assert len(set(ratings)) > 1

    def test_prediction_error_is_bounded(self, trained_model, known_user_movie_pairs):
        """Mean absolute error against known actuals stays under 2.0.

        This is deliberately loose: it is a smoke test that the right model
        file was loaded, not a substitute for proper offline evaluation.
        """
        errors = [
            abs(trained_model.predict(p["user_id"], p["movie_id"]) - p["actual_rating"])
            for p in known_user_movie_pairs
        ]
        mae = sum(errors) / len(errors)
        assert mae < 2.0, f"MAE {mae:.3f} suggests a wrong or corrupted model"

    # =========================================================================
    # Edge Case Tests
    # =========================================================================

    def test_handles_unknown_user_gracefully(self, trained_model, unknown_users):
        """Cold-start users never raise - they fall back to the global mean."""
        for user_id in unknown_users:
            rating = trained_model.predict(user_id, "242")
            assert 1.0 <= rating <= 5.0

    def test_handles_unknown_movie_gracefully(self, trained_model, unknown_movies):
        """Cold-start movies behave the same way."""
        for movie_id in unknown_movies:
            rating = trained_model.predict("196", movie_id)
            assert 1.0 <= rating <= 5.0

    def test_handles_both_unknown(self, trained_model):
        """Both sides unknown is still answerable, not an exception."""
        rating = trained_model.predict("no_such_user", "no_such_movie")
        assert 1.0 <= rating <= 5.0

    def test_handles_very_long_ids(self, trained_model):
        """An absurdly long id is treated as unknown, not a crash."""
        rating = trained_model.predict("x" * 500, "y" * 500)
        assert 1.0 <= rating <= 5.0

    def test_handles_special_characters(self, trained_model):
        """Special characters in ids do not break the lookup."""
        rating = trained_model.predict("user@#$%", "movie!&*()")
        assert 1.0 <= rating <= 5.0
