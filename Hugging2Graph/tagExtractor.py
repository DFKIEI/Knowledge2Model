import sqlite3
import ast
from collections import Counter


conn = sqlite3.connect('huggingface.db')
cursor = conn.cursor()
cursor.execute('SELECT tags FROM Models')
rows = cursor.fetchall()

tag_counter = Counter()

for id,row in enumerate(rows):
    try:
        tags_str = row[0] 
        tags = ast.literal_eval(tags_str)

        if isinstance(tags, list):
            tag_counter.update(tags) 
        else:
            print(f"Unexpected format, not a list: {tags_str}")
    except (ValueError, SyntaxError, UnicodeEncodeError) as e:
        # Some Strings are broken from Huggingface
        print(f"Error encountered, skipping row: {e}")
        print(row)
conn.close()

ranked_tags = tag_counter.most_common()


def safe_format(tag, count):
    try:
        # return f"{tag}; {count}".encode('utf-8', errors='replace').decode('utf-8')
        return f"{tag}".encode('utf-8', errors='replace').decode('utf-8')
    except UnicodeEncodeError:
        return f"Encoding error for tag {tag}: {count}"


output_file = 'topTags.txt'
occurence = 50
with open(output_file, 'w', encoding='utf-8') as file:
    for tag, count in ranked_tags:
        if count >=occurence and len(tag)>2:
            file.write(safe_format(tag, count) + '\n')