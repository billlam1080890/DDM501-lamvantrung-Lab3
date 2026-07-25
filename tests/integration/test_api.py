"""
Integration tests for API endpoints.

These exercise the full request path: HTTP -> Pydantic validation -> model
-> response serialisation. Unlike the unit tests, nothing here is mocked.

Run tests:
    pytest tests/integration/test_api.py -v
"""


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, test_client):
        """The health endpoint is reachable."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, test_client):
        """Response carries a status field."""
        response = test_client.get("/health")
        assert "status" in response.json()

    def test_health_response_has_model_loaded_field(self, test_client):
        """Response reports whether the model is loaded."""
        response = test_client.get("/health")
        assert "model_loaded" in response.json()

    def test_health_model_loaded_is_boolean(self, test_client):
        """model_loaded is a real bool - monitoring alerts on it."""
        response = test_client.get("/health")
        assert isinstance(response.json()["model_loaded"], bool)

    def test_health_reports_healthy_when_model_present(self, test_client):
        """With the model loaded the service reports healthy."""
        body = test_client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self, test_client):
        """The root endpoint is reachable."""
        assert test_client.get("/").status_code == 200

    def test_root_contains_api_info(self, test_client):
        """Root advertises name, version, and where the docs live."""
        body = test_client.get("/").json()
        for field in ("name", "version", "description", "docs", "health"):
            assert field in body, f"missing '{field}' in root response"


class TestPredictEndpoint:
    """Tests for POST /predict."""

    # =========================================================================
    # Happy path and response structure
    # =========================================================================

    def test_predict_valid_request_returns_200(self, test_client, sample_prediction_request):
        """A valid prediction request succeeds."""
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200

    def test_predict_response_has_predicted_rating(self, test_client, sample_prediction_request):
        """Response carries the predicted rating."""
        response = test_client.post("/predict", json=sample_prediction_request)
        assert "predicted_rating" in response.json()

    def test_predict_response_has_user_id(self, test_client, sample_prediction_request):
        """The request's user_id is echoed back so responses are self-describing."""
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.json()["user_id"] == sample_prediction_request["user_id"]

    def test_predict_response_has_movie_id(self, test_client, sample_prediction_request):
        """The request's movie_id is echoed back."""
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.json()["movie_id"] == sample_prediction_request["movie_id"]

    def test_predict_response_rating_in_valid_range(self, test_client, sample_prediction_request):
        """The served rating respects the 1-5 contract."""
        rating = test_client.post("/predict", json=sample_prediction_request).json()[
            "predicted_rating"
        ]
        assert 1.0 <= rating <= 5.0

    def test_predict_response_has_model_version(self, test_client, sample_prediction_request):
        """Responses are attributable to a model version."""
        response = test_client.post("/predict", json=sample_prediction_request)
        assert "model_version" in response.json()

    # =========================================================================
    # Validation Error Tests
    # =========================================================================

    def test_predict_missing_user_id_returns_422(self, test_client):
        """A missing user_id is a validation error, not a 500."""
        response = test_client.post("/predict", json={"movie_id": "242"})
        assert response.status_code == 422

    def test_predict_missing_movie_id_returns_422(self, test_client):
        """A missing movie_id is a validation error."""
        response = test_client.post("/predict", json={"user_id": "196"})
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, test_client):
        """An empty body is rejected before reaching the model."""
        assert test_client.post("/predict", json={}).status_code == 422

    def test_predict_all_invalid_requests_return_422(
        self, test_client, invalid_prediction_requests
    ):
        """Every known-bad payload is rejected with 422, never 200 or 500."""
        for payload in invalid_prediction_requests:
            response = test_client.post("/predict", json=payload)
            assert response.status_code == 422, f"payload {payload} returned {response.status_code}"

    def test_predict_wrong_method_returns_405(self, test_client):
        """GET on a POST-only endpoint is a method error."""
        assert test_client.get("/predict").status_code == 405

    # =========================================================================
    # Unknown entity handling
    # =========================================================================

    def test_predict_unknown_user_still_succeeds(self, test_client, unknown_users):
        """Cold-start users get the global-mean fallback rather than an error."""
        for user_id in unknown_users:
            response = test_client.post("/predict", json={"user_id": user_id, "movie_id": "242"})
            assert response.status_code == 200
            assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_unknown_movie_still_succeeds(self, test_client, unknown_movies):
        """Cold-start movies behave the same way."""
        for movie_id in unknown_movies:
            response = test_client.post("/predict", json={"user_id": "196", "movie_id": movie_id})
            assert response.status_code == 200
            assert 1.0 <= response.json()["predicted_rating"] <= 5.0


class TestBatchPredictEndpoint:
    """Tests for POST /predict/batch."""

    def test_batch_returns_200(self, test_client, sample_batch_request):
        """A valid batch succeeds."""
        assert test_client.post("/predict/batch", json=sample_batch_request).status_code == 200

    def test_batch_returns_one_prediction_per_item(self, test_client, sample_batch_request):
        """Output length matches input length."""
        body = test_client.post("/predict/batch", json=sample_batch_request).json()
        assert len(body["predictions"]) == len(sample_batch_request["predictions"])

    def test_batch_total_count_matches(self, test_client, sample_batch_request):
        """total_count agrees with the payload it describes."""
        body = test_client.post("/predict/batch", json=sample_batch_request).json()
        assert body["total_count"] == len(body["predictions"])

    def test_batch_all_ratings_in_range(self, test_client, sample_batch_request):
        """Every rating in the batch respects the 1-5 contract."""
        body = test_client.post("/predict/batch", json=sample_batch_request).json()
        for item in body["predictions"]:
            assert 1.0 <= item["predicted_rating"] <= 5.0

    def test_batch_empty_list_returns_422(self, test_client):
        """An empty batch is rejected."""
        assert test_client.post("/predict/batch", json={"predictions": []}).status_code == 422

    def test_batch_over_limit_returns_422(self, test_client):
        """Batches beyond the 100-item cap are rejected."""
        payload = {"predictions": [{"user_id": str(i), "movie_id": "1"} for i in range(101)]}
        assert test_client.post("/predict/batch", json=payload).status_code == 422


class TestModelInfoEndpoint:
    """Tests for GET /model/info."""

    def test_model_info_returns_200(self, test_client):
        """The endpoint is reachable."""
        assert test_client.get("/model/info").status_code == 200

    def test_model_info_contains_version_and_type(self, test_client):
        """Model metadata is exposed for auditing which version served a request."""
        body = test_client.get("/model/info").json()
        assert "model_version" in body
        assert "model_type" in body
        assert body["is_loaded"] is True


class TestOpenAPIDocs:
    """The auto-generated documentation must actually be reachable."""

    def test_openapi_schema_available(self, test_client):
        """openapi.json is served."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        assert "paths" in response.json()

    def test_swagger_docs_available(self, test_client):
        """Swagger UI is served at /docs."""
        assert test_client.get("/docs").status_code == 200

    def test_all_endpoints_documented(self, test_client):
        """Every implemented route appears in the OpenAPI schema."""
        paths = test_client.get("/openapi.json").json()["paths"]
        for route in ("/", "/health", "/predict", "/predict/batch", "/model/info"):
            assert route in paths, f"{route} missing from OpenAPI schema"
