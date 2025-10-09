import os
import sqlite3
import torch
import shutil
from pathlib import Path
from datetime import datetime, timezone
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

# Add your HuggingFace token here
HF_TOKEN = ""# Add your token here

# Setup workspace
ROOT = Path(__file__).resolve().parent
TEMP_FOLDER = ROOT / "model_testing_workspace"
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TEMP_FOLDER)

# Connect to database
DATABASE_PATH = '../ModelStatus/huggingface2.db'
try:
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    print("Connected to database")
except Exception as e:
    print(f"Database connection failed: {e}")
    exit(1)

# Check/create health columns
cursor.execute("PRAGMA table_info(Models)")
columns = [col[1] for col in cursor.fetchall()]

if 'health_status' not in columns:
    cursor.execute("ALTER TABLE Models ADD COLUMN health_status TEXT")
if 'last_checked' not in columns:
    cursor.execute("ALTER TABLE Models ADD COLUMN last_checked TIMESTAMP")
if 'health_error' not in columns:
    cursor.execute("ALTER TABLE Models ADD COLUMN health_error TEXT")
conn.commit()

# Configuration
TEST_CONFIG = {
    "problem": "text-classification",
    "libraries_to_test": ["transformers", "sentence-transformers"],
    "batch_size": 20,
    "min_downloads": 15,
    "test_sentence": "This is a simple test to check if the model works correctly.",
}

print(f"Testing {TEST_CONFIG['libraries_to_test']} with {TEST_CONFIG['batch_size']} models per library")

def update_model_health(model_id, status, error_message=""):
    try:
        cursor.execute("""
            UPDATE Models 
            SET health_status = ?, health_error = ?, last_checked = ? 
            WHERE model_id = ?
        """, (status, error_message[:500], datetime.now(timezone.utc).isoformat(), model_id))
        conn.commit()
    except Exception as e:
        print(f"Database update failed for {model_id}: {e}")

def test_transformers_model(model_name, test_text, cache_dir):
    try:
        # Try loading without trust_remote_code first
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=str(cache_dir), 
                use_fast=True,
                token=HF_TOKEN
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                cache_dir=str(cache_dir),
                token=HF_TOKEN
            )

        except Exception as e:
            if "trust_remote_code" in str(e).lower():
                # If it needs trust_remote_code, try again with it
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=str(cache_dir), 
                    use_fast=True, 
                    trust_remote_code=True,
                    token=HF_TOKEN
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    cache_dir=str(cache_dir), 
                    trust_remote_code=True,
                    token=HF_TOKEN
                )
            else:
                raise e
        
        # Create pipeline
        classifier = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True
        )
        
        # Test it
        result = classifier(test_text)
        
        if result and len(result) > 0:
            return "OK", ""
        else:
            return "FAIL", "No output from pipeline"
            
    except Exception as e:
        error_msg = str(e)[:200]
        
        # Categorize common errors
        if "out of memory" in error_msg.lower():
            return "OOM", "Out of memory"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return "NOT_FOUND", "Model not found"
        elif "trust_remote_code" in error_msg.lower():
            return "TRUST_NEEDED", "Requires trust_remote_code"
        else:
            return "FAIL", error_msg

def test_sentence_transformers_model(model_name, test_text, cache_dir):
    try:
        # First try loading as sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                model_name, 
                cache_folder=str(cache_dir),
                token=HF_TOKEN
            )
            embeddings = model.encode([test_text])
            
            if embeddings is not None and len(embeddings) > 0:
                return "OK", ""
            else:
                return "FAIL", "No embeddings generated"
                
        except Exception as sbert_error:
            # If sentence transformers fails, try as regular transformers model
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=str(cache_dir), 
                    use_fast=True,
                    token=HF_TOKEN
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    cache_dir=str(cache_dir),
                    token=HF_TOKEN
                )

                classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1, truncation=True)
                
                result = classifier(test_text)
                
                if result and len(result) > 0:
                    return "OK", "Loaded as transformers model instead of sentence-transformers"
                else:
                    return "FAIL", "Failed as both sentence-transformers and transformers"
                    
            except Exception as transformers_error:
                return "FAIL", f"SentenceTransformers: {str(sbert_error)[:100]}, Transformers: {str(transformers_error)[:100]}"
                
    except Exception as e:
        error_msg = str(e)[:200]
        
        # Categorize common errors
        if "out of memory" in error_msg.lower():
            return "OOM", "Out of memory"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return "NOT_FOUND", "Model not found"
        else:
            return "FAIL", error_msg


def process_batch(models, library_name):
    """Runs your existing per-model logic on a single batch."""
    stats = {"OK": 0, "FAIL": 0, "OOM": 0, "NOT_FOUND": 0, "OTHER": 0}

    for i, (model_id, model_name, downloads, library) in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model_name}")
        print(f"         Downloads: {downloads:,}")

        model_cache_dir = TEMP_FOLDER / f"model_{i}_{model_id.replace('/', '_').replace('@', '_')}"
        model_cache_dir.mkdir(exist_ok=True)

        try:
            if library_name == "transformers":
                status, error = test_transformers_model(
                    model_name,
                    TEST_CONFIG["test_sentence"],
                    model_cache_dir
                )
            elif library_name == "sentence-transformers":
                status, error = test_sentence_transformers_model(
                    model_name,
                    TEST_CONFIG["test_sentence"],
                    model_cache_dir
                )
            else:
                status, error = "SKIP", f"Library {library_name} not implemented"

            # Update database
            update_model_health(model_id, status, error)

            # Update stats
            if status in stats:
                stats[status] += 1
            else:
                stats["OTHER"] += 1

            # Print result
            if status == "OK":
                print("          OK")
            elif status == "OOM":
                print("           OUT OF MEMORY")
            elif status == "NOT_FOUND":
                print("          NOT FOUND")
            else:
                print(f"         {status}: {error[:50]}...")

        except Exception as e:
            update_model_health(model_id, "ERROR", str(e))
            stats["OTHER"] += 1
            print(f"          ERROR: {str(e)[:50]}...")

        finally:
            # Clear all contents of temp folder after each model
            try:
                if TEMP_FOLDER.exists():
                    for item in TEMP_FOLDER.iterdir():
                        try:
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                        except:
                            pass  # Ignore individual file deletion errors
            except:
                pass  # Folder might not exist yet
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print()  # readability

    # Per-batch summary
    print(f"\n{library_name} Summary (this batch):")
    for status, count in stats.items():
        if count > 0:
            print(f"  {status}: {count} models")

def test_models_for_library(library_name):
    batch_num = 1
    total_tested = 0

    while True:
        # Get next batch
        query = """
            SELECT model_id, model_name, downloads, library
            FROM Models
            WHERE health_status IS NULL
              AND problem = ?
              AND library = ?
              AND downloads >= ?
            ORDER BY downloads DESC
            LIMIT ?
        """
        cursor.execute(query, (
            TEST_CONFIG["problem"],
            library_name,
            TEST_CONFIG["min_downloads"],
            TEST_CONFIG["batch_size"]
        ))
        models = cursor.fetchall()

        if not models:
            print(f"No more {library_name} models to test")
            break

        print(f"\n=== {library_name} - Batch {batch_num} ===")
        process_batch(models, library_name)

        total_tested += len(models)
        print(f"Completed {library_name} batch {batch_num}. Total tested so far: {total_tested}")
        batch_num += 1


def main():
    print("=" * 60)
    print("MODEL HEALTH CHECK - Text Classification")
    print("=" * 60)
    
    if HF_TOKEN:
        print("Using HuggingFace token for private repositories")
    else:
        print("No HF token provided - private repositories will fail")
    
    for library in TEST_CONFIG["libraries_to_test"]:
        try:
            test_models_for_library(library)
        except Exception as e:
            print(f"Error testing {library}: {e}")
    
    # Show overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    cursor.execute("""
        SELECT health_status, COUNT(*) 
        FROM Models 
        WHERE problem = 'text-classification' 
        AND health_status IS NOT NULL
        GROUP BY health_status
        ORDER BY COUNT(*) DESC
    """)
    
    total = 0
    for status, count in cursor.fetchall():
        print(f"  {status}: {count} models")
        total += count
    
    print(f"  TOTAL TESTED: {total} models")
    
    # Show untested count
    cursor.execute("""
        SELECT COUNT(*) 
        FROM Models 
        WHERE problem = 'text-classification' 
        AND library IN ('transformers', 'sentence-transformers')
        AND health_status IS NULL
        AND downloads >= ?
    """, (TEST_CONFIG["min_downloads"],))
    
    untested = cursor.fetchone()[0]
    if untested > 0:
        print(f"  REMAINING: {untested} models to test")

def cleanup():
    try:
        conn.close()
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
    except:
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup()
        print("\nDone!")