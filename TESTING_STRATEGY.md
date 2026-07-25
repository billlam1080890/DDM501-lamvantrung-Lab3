# Testing Strategy — Movie Rating Prediction API

**Course:** DDM501 — MLOps · **Student:** Lam Van Trung (25MS23336)

---

## 1. Why ML systems need more than unit tests

Traditional software fails loudly: a bug throws an exception, a test goes red. An ML system fails **quietly**. The API returns 200, the response validates, the rating is between 1 and 5 — and the number is meaningless because the wrong model file was loaded, or the training data was corrupted, or the model collapsed to predicting the same value for everyone.

None of those failures produce an exception. Each one is caught by a different tier of the pyramid below.

```
                    ┌─────────────────┐
                    │  Model (19)     │  Does it behave sensibly?
                    ├─────────────────┤
                    │  Data (21)      │  Is the input trustworthy?
                    ├─────────────────┤
                    │ Integration (31)│  Do the parts work together?
                    ├─────────────────┤
                    │   Unit (43)     │  Is each piece correct?
                    └─────────────────┘
                      114 tests total
```

---

## 2. The four tiers

### Tier 1 — Unit tests (43 tests)

`tests/unit/test_model.py` · `tests/unit/test_schemas.py`

Test one function or class in isolation. Fast (<0.1 s each), no network, no server.

| What is covered | Example |
|---|---|
| Model loading | `test_model_raises_error_for_missing_file` |
| Prediction contract | `test_predict_returns_float`, `test_predict_returns_value_in_valid_range` |
| Batch semantics | `test_predict_batch_returns_correct_length` |
| Failure modes | `test_predict_raises_when_model_not_loaded`, `test_model_raises_for_corrupt_file` |
| Singleton caching | `test_get_model_returns_same_instance` |
| Schema validation | every required field, empty/whitespace/None/too-long id, boundary ratings |

**Design note.** `test_rating_at_boundaries` asserts that exactly 1.0 and exactly 5.0 are *accepted*. Off-by-one errors at inclusive bounds are the single most common validation bug, and a test that only checks the middle of the range never finds them.

### Tier 2 — Integration tests (31 tests)

`tests/integration/test_api.py`

Exercise the full path — HTTP → Pydantic → model → JSON — with nothing mocked.

| What is covered | Example |
|---|---|
| All 5 endpoints reachable | `/`, `/health`, `/predict`, `/predict/batch`, `/model/info` |
| Response structure | every documented field present and correctly typed |
| Validation → 422, not 500 | `test_predict_all_invalid_requests_return_422` |
| Wrong method → 405 | `test_predict_wrong_method_returns_405` |
| Batch limits enforced | empty batch and 101-item batch both rejected |
| Cold start | unknown users and movies still return 200 |
| Docs actually served | `/docs`, `/openapi.json`, every route in the schema |

**Design note.** The distinction between **422 and 500** is the point of this tier. Both are "the request failed", but 422 means *the client sent something invalid* and 500 means *the service is broken*. Returning 500 for a missing field would page an on-call engineer for what is really a client bug.

### Tier 3 — Data tests (21 tests)

`tests/data/test_data_quality.py`

Validate the data, not the code. Split into schema-level and statistical checks.

| Category | What it catches |
|---|---|
| Range | ratings outside 1–5 — a corrupted export or a scale change upstream |
| Type | `"4.0"` as a string, or a bool masquerading as a number |
| Completeness | missing ids, null ratings, absent fields |
| Integrity | duplicate `(user, movie)` pairs |
| Whitespace | `" 196"` and `"196"` silently becoming two different users |
| Distribution | zero variance, single rating value, only one user or one movie |
| Output quality | NaN predictions, degenerate constant output |

**Design note.** `test_rating_standard_deviation` is the highest-value test in this tier. Every schema check can pass on a dataset where every rating is exactly 3.0 — the types are right, nothing is null, all ids are present. Only the variance check reveals that the export is broken.

### Tier 4 — Model behavioral tests (19 tests)

`tests/model/test_model_behavior.py`

Verify behaviour that accuracy metrics cannot express. A model can hold its RMSE while being unusable.

**Invariance** — output must not change when the input carries no new information:

- `test_same_input_same_output` — the same pair predicts the same value twice
- `test_batch_order_independent` — reordering a batch reorders results, it does not change them
- `test_individual_vs_batch_same_results` — batching is an optimisation, not a second code path
- `test_interleaved_predictions_do_not_interfere` — no state leaks between calls

**Directional** — output must vary as inputs vary:

- `test_different_users_different_predictions` — if every user gets the same rating, there is no personalisation happening and the model is just an item-average lookup
- `test_different_movies_different_predictions` — the mirror case
- `test_unknown_user_falls_back_to_neutral` — cold start lands near the global mean, not at an extreme

**Minimum functionality** — the simplest cases that must never break:

- known users predict successfully, predictions vary, MAE against known actuals stays under 2.0
- edge cases: unknown user, unknown movie, both unknown, 500-character ids, special characters

**Design note.** `test_different_users_different_predictions` is the test that would catch a catastrophic-but-silent regression. If a training bug produced a model that returns the item mean for every user, RMSE would degrade only slightly — MovieLens item means are decent predictors — and every other test in this suite would still pass. Only this test fails.

---

## 3. Fixtures

All fixtures live in `tests/conftest.py` and are shared across tiers.

| Fixture | Scope | Purpose |
|---|---|---|
| `test_client` | session | FastAPI `TestClient` |
| `trained_model` | session | the loaded model, built once |
| `sample_prediction_request` | function | one valid payload |
| `sample_batch_request` | function | a 3-item batch |
| `sample_ratings` | function | 7 rating records for data tests |
| `invalid_prediction_requests` | function | 6 known-bad payloads |
| `known_user_movie_pairs` | function | 5 real MovieLens pairs with actual ratings |
| `unknown_users` / `unknown_movies` | function | cold-start ids |

Session scope on the two expensive fixtures matters: the model pickle is ~5 MB, and loading it per test would turn a 0.8 s suite into a multi-minute one.

**One fix was required here.** The starter's `test_client` fixture returned `TestClient(app)` directly. Starlette only runs the application lifespan when the client is used as a **context manager**, so the startup event that loads the model never fired and every prediction request returned **503**. The fixture now yields from inside a `with` block.

---

## 4. CI/CD pipeline

`.github/workflows/ci.yml` — on push to `main`/`develop` and on every PR to `main`:

| Job | What it does | Blocking |
|---|---|---|
| `lint` | flake8, black `--check`, isort `--check-only` | yes |
| `type-check` | mypy on `app/` | yes |
| `test` | all 4 tiers on Python 3.10 **and** 3.11, then coverage with `--cov-fail-under=80` | yes |
| `model-validate` | retrains and fails if RMSE > 1.05 | yes |
| `docker-build` | builds the image and curls `/health` inside the container | yes |
| `ci-success` | single aggregated check for branch protection | yes |

`.github/workflows/cd.yml` — on a `v*` tag: re-verify tests → build and push to GHCR → GitHub Release → staging (with smoke test) → production behind a manual environment approval.

**Three deliberate choices:**

1. **The model is trained in CI, not committed.** A 5 MB binary in git bloats every clone forever, and a committed model can silently drift from the code that produced it. MovieLens is 5 MB and fits in seconds.
2. **`model-validate` is a blocking job.** Tests prove the code works; they do not prove the *model* is good. A change that degrades RMSE past 1.05 fails the build rather than merging quietly.
3. **`docker-build` curls the container.** A successful `docker build` proves the image assembles, not that it runs. The smoke test proves it actually serves traffic.

Pre-commit hooks mirror the lint job, so formatting problems surface in seconds locally instead of after a five-minute CI round trip.

---

## 5. Results

```
114 passed in 0.75s

Name              Stmts   Miss  Cover
--------------------------------------
app\__init__.py       1      0   100%
app\config.py        13      0   100%
app\main.py          51     10    80%
app\model.py         42      1    98%
app\schemas.py       30      0   100%
--------------------------------------
TOTAL               137     11    92%
```

**92% coverage**, against a required minimum of 80%.

The 11 uncovered lines are the exception handlers for infrastructure failures — model-load failure at startup, and the `except` branches that convert an unexpected model error into a 500. Reaching them requires breaking the model *after* the app has started, which needs fault injection rather than a test case. They are deliberately left uncovered; testing them would mean mocking the model into failing, which tests the mock rather than the system.

---

## 6. What this strategy does not cover

Being explicit about gaps is more useful than implying full coverage:

- **Load and performance testing.** No throughput or latency assertions — that is Lab 4's scope.
- **Concurrency.** The singleton model is never exercised under parallel requests.
- **Model drift over time.** The quality gate compares against a fixed threshold, not against the previously deployed model.
- **Security.** No authentication, authorisation, or input-fuzzing tests; the API is unauthenticated by design in this lab.

---

*Lam Van Trung — 25MS23336 — DDM501*
