import json
import os
import sys

INPUT_FILE = "benchmark_results.json"
OUTPUT_FILE = "benchmark_report.txt"


def load(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run benchmark_queries.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def report(data):
    meta = data["metadata"]
    queries = data["queries"]
    lines = []

    def out(text=""):
        lines.append(text)

    out()
    out("=" * 95)
    out("  SQL vs Neo4j - BENCHMARK REPORT")
    out("=" * 95)
    out(f"  Date       : {meta['timestamp'][:19]}")
    out(f"  SQLite DB  : {meta['sqlite_db']}")
    out(f"  Neo4j      : {meta['neo4j_uri']}")
    out(f"  Models     : {meta['total_models']:,}")
    out(f"  Runs/query : {meta['runs_per_query']}")
    out("=" * 95)

    tier_labels = {1: "Simple Lookups", 2: "Multi-Relationship", 3: "Graph Traversal"}
    for tier in [1, 2, 3]:
        tier_qs = [q for q in queries if q["tier"] == tier]
        if not tier_qs:
            continue

        out(f"\n  TIER {tier}: {tier_labels[tier]}")
        out(f"  {'ID':<8} {'Name':<40} {'SQL(ms)':<10} {'Neo4j(ms)':<10} {'Ratio':<8} {'Winner':<8} {'Rows'}")
        out("  " + "-" * 90)

        for q in tier_qs:
            s = q["sqlite"]["mean_ms"]
            n = q["neo4j"]["mean_ms"]
            r = q["speedup_ratio"]
            w = q["winner"].upper()
            match = "OK" if q["rows_match"] else f"MISMATCH ({q['sqlite']['result_count']} vs {q['neo4j']['result_count']})"
            out(f"  {q['id']:<8} {q['name'][:39]:<40} {s:<10.2f} {n:<10.2f} {r:<8.2f} {w:<8} {match}")

    out(f"\n{'=' * 95}")
    out()

    full_report = "\n".join(lines)
    print(full_report)

    with open(OUTPUT_FILE, "w") as f:
        f.write(full_report)
    print(f"Report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    data = load(INPUT_FILE)
    report(data)