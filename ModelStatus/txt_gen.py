import os
import sqlite3
import torch
import shutil
from pathlib import Path
from datetime import datetime, timezone
from huggingface_hub import HfApi
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import warnings

warnings.filterwarnings("ignore")
api = HfApi()

# Enable Hugging Face fast downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# HuggingFace token
HF_TOKEN = ""# Add your token here

# Database path
DATABASE_PATH = r"D:\GraphR\ModelStatus\huggingface2.db"

# Test configuration
TEST_CONFIG = {
    "problem": "text-generation",
    "libraries_to_test": ["transformers"],
    "batch_size": 10,
    "min_downloads": 50,
    "causal_prompt": "The future of artificial intelligence is",
    "seq2seq_prompt": "Summarize: Artificial intelligence is transforming technology.",
    "max_repo_size_gb": 50.0,
}




def setup_workspace():
    """Initialize workspace and environment."""
    ROOT = Path(__file__).resolve().parent
    TEMP_FOLDER = ROOT / "model_testing_workspace"
    TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(TEMP_FOLDER)
    return TEMP_FOLDER

def setup_database():
    """Connect to database and ensure required columns exist."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        print("Connected to database")
        
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
        return conn, cursor
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        exit(1)




def update_model_health(cursor, conn, model_id, status, error_message=""):
    """Update model health status in database."""
    try:
        cursor.execute("""
            UPDATE Models 
            SET health_status = ?, health_error = ?, last_checked = ? 
            WHERE model_id = ?
        """, (status, error_message[:500], datetime.now(timezone.utc).isoformat(), model_id))
        conn.commit()
    except Exception as e:
        print(f"Database update failed for {model_id}: {e}")

def repo_is_too_big(model_name: str, max_gb: float) -> tuple[bool, float]:
    """Check if repository size exceeds limit by summing weight files."""
    try:
        
        info = api.repo_info(
            repo_id=model_name,
            repo_type="model",
            files_metadata=True,
            token=HF_TOKEN
        )
        
        bytes_total = 0
        for f in info.siblings:
            name = getattr(f, "rfilename", "") or getattr(f, "path", "")
            size = getattr(f, "size", 0) or 0
            if name.endswith((".safetensors", ".bin", ".gguf")):
                bytes_total += size
                
        size_gb = bytes_total / (1024 ** 3)
        return (size_gb >= max_gb, size_gb)
    except Exception:
        return (False, 0.0)

def cleanup_temp_folder(temp_folder):
    """Clean all contents of temporary folder."""
    try:
        if temp_folder.exists():
            for item in temp_folder.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except:
                    pass  # Ignore individual file deletion errors
    except:
        pass

def clear_gpu_memory():
    """Clear GPU memory if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




def load_model_and_tokenizer(model_name, cache_dir):
    """Load model and tokenizer, trying causal LM first, then seq2seq."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=str(cache_dir), 
        use_fast=True,
        token=HF_TOKEN
    )
    
    model = None
    model_type = None
    
    # Try causal LM first (GPT-style)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=str(cache_dir),
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model_type = "causal"
    except Exception as causal_error:
        # Try seq2seq (T5/BART-style)
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, 
                cache_dir=str(cache_dir),
                token=HF_TOKEN,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model_type = "seq2seq"
        except Exception as seq2seq_error:
            
            if "gated" in str(causal_error).lower() or "access" in str(causal_error).lower():
                raise Exception(str(causal_error))
            elif "gated" in str(seq2seq_error).lower() or "access" in str(seq2seq_error).lower():
                raise Exception(str(seq2seq_error))
            else:
                raise Exception(f"Causal: {str(causal_error)[:100]} | Seq2Seq: {str(seq2seq_error)[:100]}")
    
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"          Device: {next(model.parameters()).device}")
    else:
        print("          Device: CPU (no GPU available)")
    
    return model, tokenizer, model_type

def test_model_generation(model, tokenizer, model_type):
    """
    Runs a tiny generation smoke test with safe PAD/EOS handling for causal LMs.
    Returns: (success: bool, message: str)
    """
    use_cuda = torch.cuda.is_available()
    device_arg = 0 if use_cuda else -1

    if model_type == "seq2seq":
        # Seq2seq models already define PAD/EOS; no special handling needed.
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_arg
        )
        prompt = TEST_CONFIG["seq2seq_prompt"]
        out = generator(prompt, max_new_tokens=16, do_sample=False)
        ok = bool(out and out[0].get("generated_text", "").strip())
        return (True, f"Model type: {model_type}") if ok else (False, "No valid text generated")

    else:
        # PAD/EOS SAFETY FOR CAUSAL LMs 
        try:
            # Ensure a valid pad token exists
            if tokenizer.pad_token_id is None:
                if getattr(tokenizer, "eos_token", None) is not None:
                    # Reuse EOS as PAD (common for LLaMA/Falcon/GPT-J families)
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    # If no EOS either, add a dedicated [PAD] token
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    if hasattr(model, "resize_token_embeddings"):
                        model.resize_token_embeddings(len(tokenizer))

            # Keep model configs consistent to avoid "pad_token_id ignored" warnings
            if getattr(model, "config", None) is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
        except Exception as e:
            return False, f"PAD/EOS setup failed: {str(e)[:120]}"

        # Now run a minimal deterministic generation
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_arg
        )
        prompt = TEST_CONFIG["causal_prompt"]
        out = generator(
            prompt,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id  # safe: guaranteed to exist above
        )
        ok = bool(out and len(out[0].get("generated_text", "")) > len(prompt))
        return (True, f"Model type: {model_type}") if ok else (False, "No valid text generated")


def test_transformers_model(model_name, cache_dir):
    """Test a transformers model for text generation."""
    try:
        model, tokenizer, model_type = load_model_and_tokenizer(model_name, cache_dir)
        success, message = test_model_generation(model, tokenizer, model_type)
        
        if success:
            return "OK", message
        else:
            return "FAIL", message
            
    except Exception as e:
        error_msg = str(e)[:300]
        
        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            return "OOM", "Out of memory"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return "NOT_FOUND", "Model not found"
        elif "gated repo" in error_msg.lower() or "access" in error_msg.lower():
            return "GATED", "Gated repository - needs permission"
        elif "trust_remote_code" in error_msg.lower():
            return "TRUST_NEEDED", "Requires trust_remote_code"
        elif "contains custom code" in error_msg.lower():
            return "CUSTOM_CODE", "Contains custom code"
        else:
            return "FAIL", error_msg



def process_single_model(model_data, index, total, temp_folder, cursor, conn):
    """Process a single model."""
    model_id, model_name, downloads, library = model_data
    
    print(f"[{index}/{total}] {model_name}")
    print(f"         Downloads: {downloads:,}")
    
    # Check repository size
    too_big, size_gb = repo_is_too_big(model_name, TEST_CONFIG["max_repo_size_gb"])
    if too_big:
        update_model_health(cursor, conn, model_id, "SKIPPED_TOO_BIG", f"Repo≈{size_gb:.1f} GB")
        print(f"          SKIPPED_TOO_BIG (~{size_gb:.1f} GB)")
        clear_gpu_memory()
        print()
        return "SKIPPED_TOO_BIG"
    
    # Create model cache directory
    model_cache_dir = temp_folder / f"model_{index}_{model_id.replace('/', '_').replace('@', '_')}"
    model_cache_dir.mkdir(exist_ok=True)
    
    try:
        # Test the model
        status, error = test_transformers_model(model_name, model_cache_dir)
        
        # Update database
        update_model_health(cursor, conn, model_id, status, error)
        
        # Print result
        if status == "OK":
            print("          OK")
        elif status == "OOM":
            print("          OUT OF MEMORY")
        elif status == "NOT_FOUND":
            print("          NOT FOUND")
        elif status == "GATED":
            print("          GATED REPOSITORY")
        elif status == "CUSTOM_CODE":
            print("          CUSTOM CODE")
        else:
            print(f"          {status}: {error[:50]}...")
            
        return status
        
    except Exception as e:
        update_model_health(cursor, conn, model_id, "ERROR", str(e))
        print(f"          ERROR: {str(e)[:50]}...")
        return "ERROR"
    
    finally:
        clear_gpu_memory()
        print()


def get_models_to_test(cursor, library_name):
    """Get all models that need testing."""
    query = """
        SELECT model_id, model_name, downloads, library
        FROM Models
        WHERE health_status IS NULL
          AND problem = ?
          AND library = ?
          AND downloads >= ?
        ORDER BY downloads DESC
    """
    cursor.execute(query, (
        TEST_CONFIG["problem"],
        library_name,
        TEST_CONFIG["min_downloads"]
    ))
    return cursor.fetchall()


def test_models_for_library(library_name, temp_folder, cursor, conn):
    """Test all models for a specific library."""
    
    # Get all models at once
    models = get_models_to_test(cursor, library_name)
    
    if not models:
        print(f"\nNo {library_name} models to test")
        return
    
    total = len(models)
    print(f"\n{'='*60}")
    print(f"Testing {library_name}: {total} models found")
    print(f"{'='*60}\n")
    
    # Track statistics
    stats = {
        "OK": 0, "FAIL": 0, "OOM": 0, "NOT_FOUND": 0, 
        "TRUST_NEEDED": 0, "GATED": 0, "CUSTOM_CODE": 0, 
        "SKIPPED_TOO_BIG": 0, "ERROR": 0, "OTHER": 0
    }
    
    # Process each model sequentially
    for i, model_data in enumerate(models, 1):
        status = process_single_model(model_data, i, total, temp_folder, cursor, conn)
        
        # Update stats
        if status in stats:
            stats[status] += 1
        else:
            stats["OTHER"] += 1
    
    # Print library summary
    print(f"\n{'='*60}")
    print(f"{library_name} - Final Summary")
    print(f"{'='*60}")
    for status, count in stats.items():
        if count > 0:
            print(f"  {status}: {count} models")
    print()


def print_overall_summary(cursor):
    """Print overall testing summary."""
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    cursor.execute("""
        SELECT health_status, COUNT(*) 
        FROM Models 
        WHERE problem = 'text-generation' 
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
        WHERE problem = 'text-generation' 
        AND library IN ('transformers')
        AND health_status IS NULL
        AND downloads >= ?
    """, (TEST_CONFIG["min_downloads"],))
    
    untested = cursor.fetchone()[0]
    if untested > 0:
        print(f"  REMAINING: {untested} models to test")
    print()


def main():
    """Main execution function."""
    print("=" * 60)
    print("MODEL HEALTH CHECK - Text Generation")
    print("=" * 60)
    
    if HF_TOKEN:
        print("Using HuggingFace token for private repositories")
    else:
        print("WARNING: No HF token - private repositories will fail")
    
    # Setup
    temp_folder = setup_workspace()
    conn, cursor = setup_database()
    
    try:
        # Test each library
        for library in TEST_CONFIG["libraries_to_test"]:
            try:
                test_models_for_library(library, temp_folder, cursor, conn)
            except Exception as e:
                print(f"Error testing {library}: {e}")
        
        # Show overall summary
        print_overall_summary(cursor)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        try:
            conn.close()
            shutil.rmtree(temp_folder, ignore_errors=True)
        except:
            pass
        print("Done!")


if __name__ == "__main__":
    main()