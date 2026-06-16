"""
benchmark_queries.py
====================
Run all SQL vs Neo4j benchmark queries.
Edit the config below and run: python benchmark_queries.py
Outputs: benchmark_results.json
"""

import json
import sqlite3
import statistics
import time
from datetime import datetime
from neo4j import GraphDatabase
from query_definitions import ALL_QUERIES

# ======================== CONFIG ========================
SQLITE_DB   = "./huggingface2.db"       # SQLite DB in same folder
NEO4J_URI   = "bolt://localhost:7687"   # Neo4j must be running
NEO4J_USER  = "neo4j"
NEO4J_PASS  = "12345678"
NUM_RUNS    = 5                         # runs per query for averaging
OUTPUT_FILE = "benchmark_results.json"  # output saved here
# ========================================================


def discover_param(conn, qdef):
    """Auto-discover real param values from the DB."""
    if not qdef["discover"]:
        return {}

    cursor = conn.execute(qdef["discover"])
    ptype = qdef["param_type"]

    if ptype == "model_list":
        rows = cursor.fetchall()
        return {"model_list": [r[0] for r in rows]}

    row = cursor.fetchone()
    if not row:
        return {}

    if ptype == "problem_library":
        return {"problem": row[0], "library": row[1]}

    if ptype == "tag":
        tag = row[0].split(",")[0].strip()
        return {"tag": tag}

    if ptype == "metric":
        raw = row[0].split(",")[0].strip().split("|")[0]
        if "metric:" in raw:
            raw = raw.split("metric:")[1]
        return {"metric": raw.strip()}

    # model_name, problem, etc.
    return {ptype: row[0]}


def build_sql_params(qdef, discovered):
    """Build (sql_string, params_tuple) from discovered values."""
    sql = qdef["sql"]
    ptype = qdef["param_type"]

    if ptype is None:
        return sql, ()
    if ptype == "model_name":
        return sql, (discovered.get("model_name", ""),)
    if ptype == "problem":
        return sql, (discovered.get("problem", ""),)
    if ptype == "problem_library":
        return sql, (discovered.get("problem", ""), discovered.get("library", ""))
    if ptype == "tag":
        return sql, (f"%{discovered.get('tag', '')}%",)
    if ptype == "metric":
        return sql, (f"%{discovered.get('metric', '')}%",)
    if ptype == "model_list":
        names = discovered.get("model_list", [])
        placeholders = ",".join(["?"] * len(names))
        return sql.replace("{placeholders}", placeholders), tuple(names)
    return sql, ()


def build_neo4j_params(qdef, discovered):
    """Build Cypher params dict from discovered values."""
    ptype = qdef["param_type"]
    if ptype is None:
        return {}
    if ptype == "model_name":
        return {"model_name": discovered.get("model_name", "")}
    if ptype == "problem":
        return {"problem": discovered.get("problem", "")}
    if ptype == "problem_library":
        return {"problem": discovered.get("problem", ""), "library": discovered.get("library", "")}
    if ptype == "tag":
        return {"tag": discovered.get("tag", "")}
    if ptype == "metric":
        return {"metric_name": discovered.get("metric", "")}
    if ptype == "model_list":
        return {"model_names": discovered.get("model_list", [])}
    return {}


def time_sql(conn, sql, params, runs):
    """Run SQL query multiple times, return stats."""
    latencies = []
    rows = []
    for i in range(runs):
        start = time.perf_counter()
        try:
            cursor = conn.execute(sql, params)
            result = cursor.fetchall()
            latencies.append((time.perf_counter() - start) * 1000)
            if i == 0:
                rows = [dict(r) for r in result]
        except Exception as e:
            latencies.append((time.perf_counter() - start) * 1000)
            if i == 0:
                rows = [{"error": str(e)}]
    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "stdev_ms": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "result_count": len(rows),
        "sample": rows[:3],
    }


def time_cypher(driver, cypher, params, runs):
    """Run Cypher query multiple times, return stats."""
    latencies = []
    rows = []
    with driver.session() as session:
        for i in range(runs):
            start = time.perf_counter()
            try:
                result = session.run(cypher, **params, timeout=30)
                records = [dict(r) for r in result]
                latencies.append((time.perf_counter() - start) * 1000)
                if i == 0:
                    rows = records
            except Exception as e:
                latencies.append((time.perf_counter() - start) * 1000)
                if i == 0:
                    rows = [{"error": str(e)}]
                print(f"    [Neo4j error: {str(e)[:80]}]")

    # Make JSON-safe
    safe_sample = []
    for row in rows[:3]:
        safe_sample.append({k: v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v)
                            for k, v in row.items()})
    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "stdev_ms": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "result_count": len(rows),
        "sample": safe_sample,
    }


def main():
    # Connect
    print(f"SQLite: {SQLITE_DB}")
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row

    print(f"Neo4j:  {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()

    total_models = conn.execute("SELECT COUNT(*) FROM Models").fetchone()[0]
    print(f"Total models: {total_models}\n")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "sqlite_db": SQLITE_DB,
            "neo4j_uri": NEO4J_URI,
            "runs_per_query": NUM_RUNS,
            "total_models": total_models,
        },
        "queries": []
    }

    # Header
    print(f"{'ID':<8} {'Name':<45} {'T':<3} {'SQL(ms)':<10} {'Neo4j(ms)':<10} {'Winner':<8}")
    print("-" * 90)

    for qdef in ALL_QUERIES:
        discovered = discover_param(conn, qdef)
        sql, sql_params = build_sql_params(qdef, discovered)
        neo_params = build_neo4j_params(qdef, discovered)

        sql_res = time_sql(conn, sql, sql_params, NUM_RUNS)
        neo_res = time_cypher(driver, qdef["cypher"], neo_params, NUM_RUNS)

        winner = "neo4j" if neo_res["mean_ms"] < sql_res["mean_ms"] else "sqlite"
        ratio = round(sql_res["mean_ms"] / neo_res["mean_ms"], 3) if neo_res["mean_ms"] > 0 else float("inf")

        sql_lines = len([l for l in qdef["sql"].strip().splitlines() if l.strip()])
        cypher_lines = len([l for l in qdef["cypher"].strip().splitlines() if l.strip()])

        entry = {
            "id": qdef["id"],
            "name": qdef["name"],
            "tier": qdef["tier"],
            "params_used": {k: v if not isinstance(v, list) else f"[{len(v)} items]" for k, v in discovered.items()},
            "sql_lines": sql_lines,
            "cypher_lines": cypher_lines,
            "sqlite": sql_res,
            "neo4j": neo_res,
            "winner": winner,
            "speedup_ratio": ratio,
            "rows_match": sql_res["result_count"] == neo_res["result_count"],
        }
        results["queries"].append(entry)

        print(f"{qdef['id']:<8} {qdef['name'][:44]:<45} {qdef['tier']:<3} "
              f"{sql_res['mean_ms']:<10.2f} {neo_res['mean_ms']:<10.2f} {winner.upper():<8}")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_FILE}")

    conn.close()
    driver.close()


if __name__ == "__main__":
    main()