# GEO Shopping Lab API

This directory contains the FastAPI + SQLAlchemy implementation of the
controlled GEO e-commerce experiment. PostgreSQL is the persistent study
database; SQLite remains available only for disposable demos and test suites.
The participant API intentionally excludes treatment assignments and internal
GEO bundles from `GET /api/products` and chatbot citations.

## Install and run

From this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
uvicorn app.main:app --reload --env-file .env --port 8001
```

Copy [`.env.example`](.env.example) to a local ignored `.env`, then set
`SQLALCHEMY_DATABASE_URL` and `PORT`. PostgreSQL URLs written as `postgresql://...` are
accepted and automatically use the installed psycopg v3 driver. For example:

```powershell
$env:SQLALCHEMY_DATABASE_URL = "postgresql+psycopg://researcher:change-me@localhost:5434/GEO_DEV_DB"
uvicorn app.main:app --reload --env-file .env
```

The database schema is bootstrapped non-destructively when the API or importer
runs. Do not point a live study at an unreviewed schema change; introduce a
versioned migration before evolving a populated production schema.

## Persistent Amazon catalog import

The backend loads `backend/.env` using a small built-in parser; no dotenv
package is required. Start from [`.env.example`](.env.example). Existing
operating-system or CI environment variables take precedence, and `.env` is
ignored by Git.

Set the local source path and batch size:

```dotenv
GEO_AMAZON_CATALOG_CSV=../E-Commerce_Dataset/Amazon-Products.csv
GEO_CATALOG_IMPORT_BATCH_SIZE=1000
```

Validate the source without writing data, then import it explicitly:

```powershell
python -m app.scripts.import_amazon_catalog --dry-run
python -m app.scripts.import_amazon_catalog
```

Only `E-Commerce_Dataset/Amazon-Products.csv` is imported. Do not import the
category shard files as well: they repeat the same source catalog. The command
streams and commits each batch, uses canonical Amazon source URLs as durable
idempotency keys, and can be rerun after an interruption without overwriting
existing study rows. It stores the supplied name, category hierarchy, image,
source URL, rating, review count, discounted/list prices, INR currency, source
row, quality flags, and import timestamp. The source does not provide product
descriptions, brands, features, availability, shipping, or return policies;
the importer leaves those fields blank rather than inventing claims.

The importer creates a deterministic, category-stratified CONTROL/GEO split
and generates a fact-linked GEO bundle only for the GEO arm. This is a
catalog-level allocation of distinct source listings. For a causal same-SKU
page-content experiment, materialise matched CONTROL and GEO variants from a
shared source product and preregister the exposure assignment before recruiting.

The source dataset is available through [Kaggle's Amazon Products dataset](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset).
Review its current licence and Amazon's applicable terms before using source
images or links beyond the approved research setting.

## Local illustration and demo seeding

The configured example CSV is read only when `products` is empty, so it cannot
replace an existing catalog. See
[`../examples/real_world_commuter_bottles/README.md`](../examples/real_world_commuter_bottles/README.md)
for its manufacturer sources and design limitations. PostgreSQL does **not**
receive fictional demo products automatically; the demo is retained for
SQLite-backed local development and automated tests.

Set `GEO_CORS_ORIGINS` to a comma-separated list of allowed frontend origins
(the default permits Vite on localhost/127.0.0.1 port 5173).

## API contract

Participant endpoints:

- `GET /api/health`, `GET /api/config`, `GET /api/products`
- `POST /api/sessions`, `POST /api/assistant/query`
- `POST /api/events`, `POST /api/surveys`

Research endpoints:

- `GET /api/research/products`, `GET /api/dashboard`
- `GET /api/export/{products|sessions|queries|candidates|probes|probe_candidates|events|surveys}`
- `POST /api/admin/probes`, `POST /api/admin/products/import`

The deterministic fallback retrieval implementation is used if optional ML and
agent modules are absent. When available, `app.services.geo_service.GEOService`
is loaded dynamically; its output is normalised and logged using the same study
records.

`GET /api/products` is paginated (`limit`, `offset`, optional `category` and
`q`). Participant and probe retrieval first select a bounded SQL lexical pool,
then rank and log that exact pool; they never materialise the full catalog in
application memory.

## Tests

```powershell
py -m pytest tests -q
```

## Benchmark evaluation runner

The research-only benchmark package evaluates authorised JSON, JSONL, or CSV
exports for E-GEO, AutoGEO E-commerce, OPR-Bench, ACES, and the wider GEO
framework. It does not download benchmark datasets or invoke external answer
engines. List profiles with:

```powershell
python -m app.benchmarks catalog
```

Write a tiny fictional smoke-test fixture, then evaluate it:

```powershell
python -m app.benchmarks fixture --benchmark egeo --directory C:\Research\geo-fixture
python -m app.benchmarks evaluate --benchmark egeo `
  --cases C:\Research\geo-fixture\egeo-fixture-cases.json `
  --predictions C:\Research\geo-fixture\egeo-fixture-predictions.jsonl `
  --traces C:\Research\geo-fixture\egeo-fixture-traces.jsonl `
  --baseline CONTROL `
  --output C:\Research\geo-fixture\report.json `
  --markdown C:\Research\geo-fixture\report.md
```

The fixture is strictly for checking the pipeline. For a research run, preserve
the authorised release/version/split, candidate set, provider/model/prompt,
seed, locale, and raw output in the imported manifest. See
[`../docs/BENCHMARK_EVALUATION.md`](../docs/BENCHMARK_EVALUATION.md) for the
canonical input schema, metric definitions, and interpretation constraints.

Tests use a temporary SQLite database and verify the public experiment contract.
For a human-subject study, add authentication for researcher routes, a consent
and withdrawal workflow, institutional privacy controls, and a versioned
research data-management process before deployment.
