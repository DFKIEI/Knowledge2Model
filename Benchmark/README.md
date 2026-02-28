# Benchmark: SQLite vs Neo4j

Performance and complexity comparison of equivalent queries executed against a SQLite database and a Neo4j graph database, both populated with Hugging Face model data.

---

## Files

| File | Role |
|------|------|
| `query_definitions.py` | Library of all benchmark queries (SQL + Cypher pairs) |
| `benchmark_queries.py` | Runs every query on both databases and saves timing results |
| `benchmark_report.py` | Reads the timing results and prints/saves a formatted report |

---

## Prerequisites

- SQLite database file `huggingface2.db` present in the `Benchmark/` folder
- Neo4j running locally (default: `bolt://localhost:7687`)
---

## How to Run

### Step 1 — Run the benchmark

```bash
cd Benchmark
python benchmark_queries.py
```

This connects to both databases, executes all 13 queries 5 times each, and writes the timing results to `benchmark_results.json`.

**Config** (edit at the top of `benchmark_queries.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SQLITE_DB` | `./huggingface2.db` | Path to the SQLite database |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASS` | `<your-password>` | Neo4j password |
| `NUM_RUNS` | `5` | Number of timed runs per query (results are averaged) |
| `OUTPUT_FILE` | `benchmark_results.json` | Where results are saved |

### Step 2 — Generate the report

```bash
python benchmark_report.py
```

Reads `benchmark_results.json` and produces a formatted report both in the terminal and saved to `benchmark_report.txt`.

> **Note:** Step 2 requires `benchmark_results.json` to exist. Always run Step 1 first.

---

## What Each File Does

### `query_definitions.py`

Defines all benchmark queries as a list of dictionaries. Queries are grouped into three complexity tiers:

| Tier | Category | Queries |
|------|----------|---------|
| 1 | Simple Lookups | T1_01 – T1_04 |
| 2 | Multi-Relationship | T2_01 – T2_05 |
| 3 | Graph Traversal | T3_01, T3_03 – T3_05 |


The module exports `ALL_QUERIES = TIER_1 + TIER_2 + TIER_3`.

**Query index:**

| ID | Name |
|----|------|
| T1_01 | Single model lookup by name |
| T1_02 | Top 20 models by downloads |
| T1_03 | Filter models by problem type |
| T1_04 | Count models per library |
| T2_01 | Models by problem + library |
| T2_02 | Models with specific tag + healthy status |
| T2_03 | Full context for a problem type |
| T2_04 | Models evaluated by a specific metric |
| T2_05 | Cross-library comparison for a problem |
| T3_01 | Find similar models (shared tags >= 2) |
| T3_03 | Multi-hop: Problem -> Models -> Metrics |
| T3_04 | Subgraph extraction (RAG use case) |
| T3_05 | Problems sharing models via common libraries |

---

### `benchmark_queries.py`

The main benchmark runner. Its responsibilities:

1. **Connects** to both SQLite (`sqlite3`) and Neo4j (`neo4j` driver).
2. **Parameter discovery** — for each query, runs the `discover` SQL against SQLite to find a real, representative parameter value. This avoids hard-coded test values and makes the benchmark reflect actual data.
3. **Timed execution** — runs each query `NUM_RUNS` times on both backends using `time.perf_counter()`, collecting mean, median, stdev, min, and max latency in milliseconds.
4. **Winner determination** — compares mean latency; the faster backend wins. Computes a speed ratio (`sql_ms / neo4j_ms`).
5. **Row-count verification** — checks whether both backends returned the same number of rows.
6. **Saves** the full result (timings, samples, metadata) to `benchmark_results.json`.

---

### `benchmark_report.py`

Reads `benchmark_results.json` and produces a structured human-readable report.

The report is printed to the terminal and also saved to `benchmark_report.txt`.

---

