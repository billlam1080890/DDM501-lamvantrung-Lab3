# Lab 3: Testing & CI/CD for ML Systems — DDM501

![CI](https://github.com/billlam1080890/DDM501-lamvantrung-Lab3/actions/workflows/ci.yml/badge.svg)

**Course:** DDM501 — MLOps
**Instructor:** Dr. Huynh Cong Viet Ngu
**Student:** Lam Van Trung
**Student ID:** 25MS23336

A four-tier test suite and a full CI/CD pipeline for the movie rating prediction API from [Lab 1](https://github.com/billlam1080890/DDM501-lamvantrung-Lab1).

---

## Results

```
114 passed in 0.75s          coverage: 92%  (required: 80%)
```

| Tier | Directory | Tests | What it protects against |
|---|---|---|---|
| Unit | `tests/unit/` | 43 | Broken functions, invalid schemas, unhandled failure modes |
| Integration | `tests/integration/` | 31 | Endpoints that break when wired together; 500s where 422s belong |
| Data | `tests/data/` | 21 | Corrupted, incomplete, or degenerate input data |
| Model | `tests/model/` | 19 | Non-deterministic, flat, or nonsensical model behaviour |

Full rationale for each tier: **[TESTING_STRATEGY.md](TESTING_STRATEGY.md)**

Coverage by module:

| Module | Statements | Coverage |
|---|---|---|
| `app/schemas.py` | 30 | 100% |
| `app/config.py` | 13 | 100% |
| `app/model.py` | 42 | 98% |
| `app/main.py` | 51 | 80% |
| **Total** | **137** | **92%** |

---

## Quick start

```bash
python -m venv venv
venv\Scripts\activate                       # Windows
pip install -r requirements.txt -r requirements-dev.txt

python scripts/train_model.py               # produces models/svd_model.pkl
pytest tests/ -v --cov=app --cov-report=term-missing
```

Run one tier at a time:

```bash
pytest tests/unit/ -v          # 43 tests, <1s
pytest tests/integration/ -v   # 31 tests
pytest tests/data/ -v          # 21 tests
pytest tests/model/ -v         # 19 tests
```

Enable the local hooks (they mirror the CI lint job):

```bash
pre-commit install
pre-commit run --all-files
```

---

## CI/CD

### CI — `.github/workflows/ci.yml`

Runs on every push to `main`/`develop` and every PR to `main`.

```
lint ──────┐
           ├──> test (3.10, 3.11) ──┬──> model-validate ──┐
type-check ┘                        └──> docker-build ────┴──> ci-success
```

| Job | Gate |
|---|---|
| `lint` | flake8 + black `--check` + isort `--check-only` |
| `type-check` | mypy on `app/` |
| `test` | all 4 tiers on two Python versions, then `--cov-fail-under=80` |
| `model-validate` | retrains and **fails if RMSE > 1.05** |
| `docker-build` | builds the image, then curls `/health` inside the running container |
| `ci-success` | one aggregated status check for branch protection |

### CD — `.github/workflows/cd.yml`

Triggered by a `v*` tag:

```
verify tests → build & push to GHCR → GitHub Release → staging (+smoke test) → production (manual approval)
```

Publishing goes to **GHCR** rather than Docker Hub so the workflow needs no external secrets — `GITHUB_TOKEN` is issued to every run automatically.

```bash
git tag -a v1.0.0 -m "First release"
git push origin v1.0.0
```

---

## Three decisions worth explaining

**The model is trained in CI, not committed to git.** A 5 MB pickle in version control bloats every clone permanently, and a committed model can silently drift away from the code that produced it. MovieLens 100K downloads and fits in seconds, so CI rebuilds it every run.

**`model-validate` is a blocking job.** Passing tests prove the *code* works; they say nothing about whether the *model* is any good. A change that pushes RMSE past 1.05 fails the build instead of merging quietly.

**`docker-build` smoke-tests the container.** A green `docker build` proves the image assembles, not that it starts. The job runs the container and polls `/health` until it answers — that is the check that catches a missing dependency or a bad entrypoint.

---

## Project structure

```
DDM501-lamvantrung-Lab3/
├── app/                        # Application (from Lab 1)
│   ├── main.py                 # 5 endpoints
│   ├── model.py                # MovieRatingModel wrapper
│   ├── schemas.py              # Pydantic request/response models
│   └── config.py
├── tests/
│   ├── conftest.py             # 9 shared fixtures
│   ├── unit/                   # 43 tests
│   ├── integration/            # 31 tests
│   ├── data/                   # 21 tests
│   └── model/                  # 19 tests
├── .github/workflows/
│   ├── ci.yml                  # 6 jobs
│   └── cd.yml                  # tag-triggered release pipeline
├── .pre-commit-config.yaml     # 7 hook groups
├── pyproject.toml              # black / isort / mypy / pytest config
├── TESTING_STRATEGY.md         # testing strategy document
└── scripts/train_model.py
```

---

## Fixes applied to the starter

Three changes were required to make the suite pass and the pipeline run:

| Fix | Why it was necessary |
|---|---|
| `conftest.py`: `TestClient` used as a context manager | Starlette runs the app lifespan only inside a `with` block. Without it the startup event never fired, the model stayed `None`, and **14 integration tests failed with 503** |
| `scikit-surprise` 1.1.3 → 1.1.5 | 1.1.3 has no Python 3.11 wheel and its C extension fails to build against modern numpy headers, breaking both local installs and the CI runner |
| `app/main.py`: `model: Optional[MovieRatingModel]` | The original `model: MovieRatingModel = None` is a type error — mypy rejects it, and the `type-check` job would fail |

Source files were also formatted with black and isort so the `lint` job passes on a clean checkout.

---

*Lam Van Trung — 25MS23336 — DDM501*
