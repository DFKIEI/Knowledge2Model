import os
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ===== Config =====
HF_TOKEN = os.getenv("HF_TOKEN", "") # Path to your Hugging Face token
DATABASE_PATH = r"" # Path to your database
PROBLEM = "summarization"
SUPPORTED_LIB = "transformers"
MAX_REPO_SIZE_GB = 40.0  # hard cap
MIN_DOWNLOADS = 15 

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "model_testing_workspace_summarization"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_ROOT)

# ===== DB =====
conn = sqlite3.connect(DATABASE_PATH)
cur = conn.cursor()

def ensure_columns():
    cur.execute("PRAGMA table_info(Models)")
    cols = {c[1] for c in cur.fetchall()}
    if "health_status" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN health_status TEXT")
    if "health_error" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN health_error TEXT")
    if "last_checked" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN last_checked TIMESTAMP")
    conn.commit()

def update_health(model_id: str, status: str, error: str = ""):
    cur.execute(
        """UPDATE Models
              SET health_status=?, health_error=?, last_checked=?
            WHERE model_id=?""",
        (status, (error or "")[:500], datetime.now(timezone.utc).isoformat(), model_id),
    )
    conn.commit()

# ===== HF repo size helper =====
def repo_size_gb(model_name: str) -> float:
    """Sum HF sibling file sizes; return 0.0 if lookup fails (don't block)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_name, token=HF_TOKEN)
        total = 0
        for s in (getattr(info, "siblings", None) or []):
            size = getattr(s, "size", None)
            if size is not None:
                try:
                    total += int(size)
                except Exception:
                    pass
        return total / (1024 ** 3)
    except Exception:
        return 0.0

# ===== Test payload =====
TEST_TEXT = (
    "Large language models (LLMs) are capable of a wide range of tasks including summarization, "
    "question answering, and translation. This short paragraph exists only to provide a minimal "
    "input for a health check that ensures the model can run inference and produce a non-empty summary."
)

# ===== Single-model test =====
def try_transformers_summarization(model_name: str):
    # size gate BEFORE any download
    size_gb = repo_size_gb(model_name)
    if size_gb and size_gb > MAX_REPO_SIZE_GB:
        return "SKIP_LARGE", f"Repo {size_gb:.2f} GB > {MAX_REPO_SIZE_GB:.2f} GB"

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1

    def ok(out):
        # Accept list[{'summary_text' or 'generated_text'}], list[str], or str
        if isinstance(out, str) and out.strip():
            return True
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                txt = first.get("summary_text") or first.get("generated_text") or ""
                return isinstance(txt, str) and txt.strip()
            if isinstance(first, str):
                return first.strip() != ""
        return False

    try:
        pipe = pipeline(
            "summarization",
            model=model_name,
            device=device,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        out = pipe(TEST_TEXT, max_new_tokens=64)
        if ok(out):
            return "OK", ""
    except RuntimeError as e:
        # Retry on CPU if CUDA OOM
        if "out of memory" in str(e).lower() and device == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            pipe = pipeline(
                "summarization",
                model=model_name,
                device=-1,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            out = pipe(TEST_TEXT, max_new_tokens=64)
            if ok(out):
                return "OK", ""
        else:
            raise
    except Exception:
        # Some models expose only text-generation interface; try generic pipeline as fallback
        try:
            from transformers import pipeline as gen_pipe
            gen = gen_pipe(
                "text2text-generation",
                model=model_name,
                device=(0 if torch.cuda.is_available() else -1),
                token=HF_TOKEN,
                trust_remote_code=True
            )
            out = gen("Summarize: " + TEST_TEXT, max_new_tokens=64)
            if ok(out):
                return "OK", ""
        except Exception:
            pass

    return "FAIL", "No summary produced."

# ===== FS utils =====
def clean_dir(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

# ===== Sequential fetch-next =====
def fetch_next():
    cur.execute("""
        SELECT model_id, model_name, downloads, library
          FROM Models
         WHERE problem=? AND library=?
           AND health_status IS NULL
           AND downloads >= ?
         ORDER BY downloads DESC
         LIMIT 1
    """, (PROBLEM, SUPPORTED_LIB, MIN_DOWNLOADS))
    return cur.fetchone()

def count_remaining():
    cur.execute("""
        SELECT COUNT(*) FROM Models
         WHERE problem=? AND library=?
           AND health_status IS NULL
           AND downloads >= ?
    """, (PROBLEM, SUPPORTED_LIB, MIN_DOWNLOADS))
    return cur.fetchone()[0]

def main():
    ensure_columns()

    remaining = count_remaining()
    print(f"Starting sequential health check for '{PROBLEM}' | remaining ({SUPPORTED_LIB}): {remaining}")

    i = 0
    while True:
        row = fetch_next()
        if not row:
            print("No more models to test.")
            break

        model_id, model_name, downloads, library = row
        i += 1
        print(f"[{i}] Testing: {model_name} | downloads: {downloads:,} | lib: {library}")

        model_cache = CACHE_ROOT / f"{model_id.replace('/', '_').replace('@', '_')}"
        model_cache.mkdir(parents=True, exist_ok=True)

        try:
            status, err = try_transformers_summarization(model_name)
            update_health(model_id, status, err)
            print("   ->", "OK" if status == "OK" else f"{status}: {err[:160]}")
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg:
                update_health(model_id, "NOT_FOUND", str(e))
            elif "out of memory" in msg:
                update_health(model_id, "OOM", "Out of memory")
            elif "is not a supported task" in msg:
                update_health(model_id, "FAIL", "Task unsupported by model")
            elif "trust_remote_code" in msg:
                update_health(model_id, "TRUST_NEEDED", "Requires trust_remote_code")
            else:
                update_health(model_id, "FAIL", str(e)[:500])
            print("   -> ERROR:", str(e)[:200])
        finally:
            clean_dir(model_cache)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print("\nSummary (updated rows):")
    cur.execute("""
        SELECT health_status, COUNT(*)
          FROM Models
         WHERE problem=?
           AND health_status IS NOT NULL
    """, (PROBLEM,))
    for status, cnt in cur.fetchall():
        print(f"  {status}: {cnt}")

if __name__ == "__main__":
    try:
        main()
    finally:
        conn.close()
