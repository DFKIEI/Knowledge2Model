"""
query_definitions.py
Paired SQL and Cypher queries organized by complexity tier.
"""

# Tier 1 — Simple lookups
TIER_1 = [
    {
        "id": "T1_01",
        "name": "Single model lookup by name",
        "tier": 1,
        "sql": """
            SELECT model_id, model_name, problem, library, downloads, likes,
                   health_status, last_checked, health_error
            FROM Models WHERE model_name = ?
        """,
        "cypher": """
            MATCH (m:Model {name: $model_name})
            OPTIONAL MATCH (m)-[:HAS_PROBLEM]->(p:Problem)
            OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
            RETURN m.id AS model_id, m.name AS model_name,
                   p.name AS problem, l.name AS library,
                   m.downloads AS downloads, m.likes AS likes,
                   m.health_status AS health_status
        """,
        "discover": "SELECT model_name FROM Models WHERE model_name IS NOT NULL LIMIT 1",
        "param_type": "model_name",
    },
    {
        "id": "T1_02",
        "name": "Top 20 models by downloads",
        "tier": 1,
        "sql": """
            SELECT model_name, problem, library, downloads, likes, health_status
            FROM Models ORDER BY downloads DESC LIMIT 20
        """,
        "cypher": """
            MATCH (m:Model)
            OPTIONAL MATCH (m)-[:HAS_PROBLEM]->(p:Problem)
            OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
            RETURN m.name AS model_name, p.name AS problem,
                   l.name AS library, m.downloads AS downloads,
                   m.likes AS likes, m.health_status AS health_status
            ORDER BY m.downloads DESC LIMIT 20
        """,
        "discover": None,
        "param_type": None,
    },
    {
        "id": "T1_03",
        "name": "Filter models by problem type",
        "tier": 1,
        "sql": """
            SELECT model_name, library, downloads, likes, health_status
            FROM Models WHERE problem = ? ORDER BY downloads DESC LIMIT 50
        """,
        "cypher": """
            MATCH (m:Model)-[:HAS_PROBLEM]->(p:Problem {name: $problem})
            OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
            RETURN m.name AS model_name, l.name AS library,
                   m.downloads AS downloads, m.likes AS likes,
                   m.health_status AS health_status
            ORDER BY m.downloads DESC LIMIT 50
        """,
        "discover": "SELECT problem FROM Models WHERE problem IS NOT NULL AND problem != '' GROUP BY problem ORDER BY COUNT(*) DESC LIMIT 1",
        "param_type": "problem",
    },
    {
        "id": "T1_04",
        "name": "Count models per library",
        "tier": 1,
        "sql": """
            SELECT library, COUNT(*) as model_count
            FROM Models WHERE library IS NOT NULL AND library != ''
            GROUP BY library ORDER BY model_count DESC LIMIT 20
        """,
        "cypher": """
            MATCH (m:Model)-[:USES_LIBRARY]->(l:Library)
            WHERE l.name IS NOT NULL AND l.name <> ''
            RETURN l.name AS library, COUNT(m) AS model_count
            ORDER BY model_count DESC LIMIT 20
        """,
        "discover": None,
        "param_type": None,
    },
]

# Tier 2 — Multi-relationship queries
TIER_2 = [
    {
        "id": "T2_01",
        "name": "Models by problem + library",
        "tier": 2,
        "sql": """
            SELECT model_name, downloads, likes, health_status
            FROM Models WHERE problem = ? AND library = ?
            ORDER BY downloads DESC LIMIT 20
        """,
        "cypher": """
            MATCH (m:Model)-[:HAS_PROBLEM]->(p:Problem {name: $problem})
            MATCH (m)-[:USES_LIBRARY]->(l:Library {name: $library})
            RETURN m.name AS model_name, m.downloads AS downloads,
                   m.likes AS likes, m.health_status AS health_status
            ORDER BY m.downloads DESC LIMIT 20
        """,
        "discover": "SELECT problem, library FROM Models WHERE problem IS NOT NULL AND problem != '' AND library IS NOT NULL AND library != '' GROUP BY problem, library ORDER BY COUNT(*) DESC LIMIT 1",
        "param_type": "problem_library",
    },
    {
        "id": "T2_02",
        "name": "Models with specific tag + healthy status",
        "tier": 2,
        "sql": """
            SELECT model_name, problem, library, downloads, health_status
            FROM Models WHERE model_card_tags LIKE ? AND health_status = 'OK'
            ORDER BY downloads DESC LIMIT 20
        """,
        "cypher": """
            MATCH (m:Model)-[:HAS_TAG]->(t:Tag {name: $tag})
            WHERE m.health_status = 'OK'
            OPTIONAL MATCH (m)-[:HAS_PROBLEM]->(p:Problem)
            OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
            RETURN m.name AS model_name, p.name AS problem,
                   l.name AS library, m.downloads AS downloads,
                   m.health_status AS health_status
            ORDER BY m.downloads DESC LIMIT 20
        """,
        "discover": "SELECT model_card_tags FROM Models WHERE model_card_tags IS NOT NULL AND model_card_tags != '' AND health_status = 'OK' LIMIT 1",
        "param_type": "tag",
    },
    {
        "id": "T2_03",
        "name": "Full context for a problem type",
        "tier": 2,
        "sql": """
            SELECT model_name, problem, library, coverTag, model_card_tags,
                   downloads, likes, health_status, last_checked, health_error
            FROM Models WHERE problem = ? ORDER BY downloads DESC LIMIT 20
        """,
        "cypher": """
            MATCH (m:Model)-[:HAS_PROBLEM]->(p:Problem {name: $problem})
            OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
            OPTIONAL MATCH (m)-[:HAS_COVER_TAG]->(ct:CoverTag)
            OPTIONAL MATCH (m)-[:HAS_TAG]->(t:Tag)
            RETURN m.name AS model_name, p.name AS problem,
                   l.name AS library, ct.name AS coverTag,
                   collect(DISTINCT t.name) AS tags,
                   m.downloads AS downloads, m.likes AS likes,
                   m.health_status AS health_status
            ORDER BY m.downloads DESC LIMIT 20
        """,
        "discover": "SELECT problem FROM Models WHERE problem IS NOT NULL AND problem != '' GROUP BY problem ORDER BY COUNT(*) DESC LIMIT 1",
        "param_type": "problem",
    },
    {
        "id": "T2_04",
        "name": "Models evaluated by a specific metric",
        "tier": 2,
        "sql": """
            SELECT model_name, problem, library, downloads, metrics
            FROM Models WHERE metrics LIKE ?
            ORDER BY downloads DESC LIMIT 20
        """,
        "cypher": """
            MATCH (m:Model)-[:EVALUATED_BY]->(met:Metric {name: $metric_name})
            OPTIONAL MATCH (m)-[:HAS_PROBLEM]->(p:Problem)
            OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
            OPTIONAL MATCH (met)-[:ON_DATASET]->(d:Dataset)
            OPTIONAL MATCH (met)-[:HAS_SCORE]->(s:Score)
            RETURN m.name AS model_name, p.name AS problem,
                   l.name AS library, m.downloads AS downloads,
                   met.name AS metric, d.name AS dataset, s.value AS score
            ORDER BY m.downloads DESC LIMIT 20
        """,
        "discover": "SELECT metrics FROM Models WHERE metrics IS NOT NULL AND metrics != '' LIMIT 1",
        "param_type": "metric",
    },
    {
        "id": "T2_05",
        "name": "Cross-library comparison for a problem",
        "tier": 2,
        "sql": """
            SELECT library, COUNT(*) as model_count,
                   AVG(downloads) as avg_downloads, SUM(downloads) as total_downloads
            FROM Models WHERE problem = ? AND library IS NOT NULL AND library != ''
            GROUP BY library ORDER BY model_count DESC
        """,
        "cypher": """
            MATCH (m:Model)-[:HAS_PROBLEM]->(p:Problem {name: $problem})
            MATCH (m)-[:USES_LIBRARY]->(l:Library)
            WHERE l.name IS NOT NULL AND l.name <> ''
            RETURN l.name AS library, COUNT(m) AS model_count,
                   AVG(m.downloads) AS avg_downloads, SUM(m.downloads) AS total_downloads
            ORDER BY model_count DESC
        """,
        "discover": "SELECT problem FROM Models WHERE problem IS NOT NULL AND problem != '' GROUP BY problem ORDER BY COUNT(*) DESC LIMIT 1",
        "param_type": "problem",
    },
]

# Tier 3 — Graph traversal (Neo4j's sweet spot)
TIER_3 = [
    {
        "id": "T3_01",
        "name": "Find similar models (shared tags >= 2)",
        "tier": 3,
        "sql": """
            WITH target AS (
                SELECT model_name, model_card_tags FROM Models WHERE model_name = ?
            )
            SELECT m.model_name, m.problem, m.library, m.downloads, m.health_status
            FROM Models m, target t
            WHERE m.model_name != t.model_name
              AND (
                  (CASE WHEN m.model_card_tags LIKE '%' || TRIM(SUBSTR(t.model_card_tags, 1, INSTR(t.model_card_tags || ',', ',') - 1)) || '%' THEN 1 ELSE 0 END) +
                  (CASE WHEN LENGTH(t.model_card_tags) - LENGTH(REPLACE(t.model_card_tags, ',', '')) >= 1
                        AND m.model_card_tags LIKE '%' || TRIM(SUBSTR(t.model_card_tags, INSTR(t.model_card_tags, ',') + 1, INSTR(SUBSTR(t.model_card_tags, INSTR(t.model_card_tags, ',') + 1) || ',', ',') - 1)) || '%'
                   THEN 1 ELSE 0 END)
              ) >= 2
            ORDER BY m.downloads DESC LIMIT 20
        """,
        "cypher": """
            MATCH (target:Model {name: $model_name})-[:HAS_TAG]->(t:Tag)<-[:HAS_TAG]-(similar:Model)
            WHERE target <> similar
            WITH similar, COUNT(DISTINCT t) AS shared_tags
            WHERE shared_tags >= 2
            OPTIONAL MATCH (similar)-[:HAS_PROBLEM]->(p:Problem)
            OPTIONAL MATCH (similar)-[:USES_LIBRARY]->(l:Library)
            RETURN similar.name AS model_name, p.name AS problem,
                   l.name AS library, similar.downloads AS downloads,
                   similar.health_status AS health_status, shared_tags
            ORDER BY shared_tags DESC, similar.downloads DESC LIMIT 20
        """,
        "discover": "SELECT model_name FROM Models WHERE model_card_tags IS NOT NULL AND LENGTH(model_card_tags) - LENGTH(REPLACE(model_card_tags, ',', '')) >= 3 ORDER BY downloads DESC LIMIT 1",
        "param_type": "model_name",
    },
    {
        "id": "T3_03",
        "name": "Multi-hop: Problem -> Models -> Metrics",
        "tier": 3,
        "sql": """
            SELECT model_name, problem, library, downloads, metrics
            FROM Models WHERE problem = ? AND metrics IS NOT NULL AND metrics != ''
            ORDER BY downloads DESC LIMIT 20
        """,
        "cypher": """
            MATCH (p:Problem {name: $problem})<-[:HAS_PROBLEM]-(m:Model)
            WITH m, p ORDER BY m.downloads DESC LIMIT 20
            OPTIONAL MATCH (m)-[:EVALUATED_BY]->(met:Metric)
            WITH m, p, collect(DISTINCT met.name) AS metric_names
            OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
            RETURN m.name AS model_name, p.name AS problem,
                   l.name AS library, m.downloads AS downloads, metric_names
            ORDER BY m.downloads DESC
        """,
        "discover": "SELECT problem FROM Models WHERE problem IS NOT NULL AND problem != '' AND metrics IS NOT NULL AND metrics != '' GROUP BY problem HAVING COUNT(*) BETWEEN 10 AND 100 ORDER BY COUNT(*) DESC LIMIT 1",
        "param_type": "problem",
    },
    {
        "id": "T3_04",
        "name": "Subgraph extraction (RAG use case)",
        "tier": 3,
        "sql": """
            SELECT model_name, problem, coverTag, library, model_card_tags,
                   downloads, likes, lastModified, metrics, health_status
            FROM Models WHERE model_name IN ({placeholders})
            ORDER BY downloads DESC
        """,
        "cypher": """
            WITH $model_names AS names
            MATCH (m:Model) WHERE m.name IN names
            OPTIONAL MATCH (m)-[:HAS_PROBLEM]->(p:Problem)
            OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
            WITH m, p, l
            OPTIONAL MATCH (m)-[:HAS_TAG]->(t:Tag)
            WITH m, p, l, collect(DISTINCT t.name) AS tags
            OPTIONAL MATCH (m)-[:HAS_COVER_TAG]->(ct:CoverTag)
            RETURN m.name AS model_name, m.downloads AS downloads, m.likes AS likes,
                   p.name AS problem, l.name AS library, tags,
                   collect(DISTINCT ct.name) AS coverTags,
                   m.health_status AS health_status
            ORDER BY m.downloads DESC
        """,
        "discover": "SELECT model_name FROM Models WHERE model_name IS NOT NULL ORDER BY downloads DESC LIMIT 10",
        "param_type": "model_list",
    },
    {
        "id": "T3_05",
        "name": "Problems sharing models via common libraries",
        "tier": 3,
        "sql": """
            SELECT m2.problem AS related_problem, m2.library,
                   COUNT(*) AS shared_model_count
            FROM Models m1 JOIN Models m2 ON m1.library = m2.library
            WHERE m1.problem = ? AND m2.problem != m1.problem
                  AND m2.problem IS NOT NULL AND m2.problem != ''
            GROUP BY m2.problem, m2.library
            ORDER BY shared_model_count DESC LIMIT 20
        """,
        "cypher": """
            MATCH (p1:Problem {name: $problem})<-[:HAS_PROBLEM]-(m1:Model)-[:USES_LIBRARY]->(l:Library)
            WITH p1, l
            MATCH (l)<-[:USES_LIBRARY]-(m2:Model)-[:HAS_PROBLEM]->(p2:Problem)
            WHERE p1 <> p2
            RETURN p2.name AS related_problem, l.name AS library,
                   COUNT(DISTINCT m2) AS shared_model_count
            ORDER BY shared_model_count DESC LIMIT 20
        """,
        "discover": "SELECT problem FROM Models WHERE problem IS NOT NULL AND problem != '' GROUP BY problem HAVING COUNT(*) BETWEEN 20 AND 100 ORDER BY COUNT(*) DESC LIMIT 1",
        "param_type": "problem",
    },
]

ALL_QUERIES = TIER_1 + TIER_2 + TIER_3