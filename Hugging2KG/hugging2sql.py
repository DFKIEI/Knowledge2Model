import sqlite3
import requests
import json

response = requests.get(
    "https://huggingface.co/api/models-tags-by-type",
    params={},
    headers={}
)

tags = json.loads(response.content)["pipeline_tag"]
# libraries = json.loads(response.content)["library"]
print(tags)


conn = sqlite3.connect('./Hugging2KG/huggingface.db')
cursor = conn.cursor()


cursor.execute('''
CREATE TABLE IF NOT EXISTS Models (
    model_id TEXT PRIMARY KEY,
    model_name TEXT,
    problem TEXT,
    tags TEXT,
    coverTag TEXT,
    library TEXT,
    downloads INTEGER,
    likes INTEGER,
    lastModified TEXT
)
''')
conn.commit()

for tag in tags:
    problem = tag["id"]
    coverTag = tag["subType"]

    print(problem)
    
    response = requests.get(
        "https://huggingface.co/api/models",
        params={"limit": "unlimited", "full": "True", "config": "True", "sort": "downloads", "filter": problem},
        headers={}
    )
    data = json.loads(response.content)

    for model in data:
        try:
            model_id = model['_id']
            model_name = model['id']
            tags = json.dumps(model['tags'])
            library = model["library_name"]
            downloads = model['downloads']
            likes = model['downloads']
            modified = model['lastModified']
        
            cursor.execute('''
                INSERT INTO Models (model_id, model_name, problem, tags, coverTag, library, downloads, likes, lastModified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (model_id, model_name, problem, tags, coverTag, library, downloads, likes, modified))
        except sqlite3.IntegrityError:
            # print(model)
            pass
        except KeyError:
            # print(model)
            pass

    conn.commit()

    cursor.execute('SELECT COUNT(*) FROM Models')
    print(cursor.fetchone()[0])


conn.close()
