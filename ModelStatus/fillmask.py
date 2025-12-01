import os
import sqlite3
import torch
import shutil
from pathlib import Path
from datetime import datetime, timezone
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import warnings
warnings.filterwarnings("ignore")

# --------------------
# Config 
# --------------------
HF_TOKEN = os.getenv("HF_TOKEN", "hf_xxx")
DATABASE_PATH = "./huggingface2.db"
PROBLEM = "fill-mask"
SUPPORTED_LIB = "transformers"
UNSUPPORTED_LIBS = {"transformers.js", "multimolecule"}  # mark as SKIP
TOP_K = int(os.getenv("FILLMASK_TOPK", "1"))
MAX_REPO_SIZE_GB = float(os.getenv("MAX_REPO_SIZE_GB", "60"))  # <— NEW

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "model_testing_workspace_fillmask"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_ROOT)

# --------------------
# DB setup
# --------------------
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
              SET health_status = ?, health_error = ?, last_checked = ?
            WHERE model_id = ?""",
        (status, (error or "")[:500], datetime.now(timezone.utc).isoformat(), model_id),
    )
    conn.commit()

# --------------------
# Size helper  <— NEW
# --------------------
def repo_size_gb(model_name: str) -> float:
    """Return total repo size in GB (sum of sibling files). If unknown, return 0.0 so we don't block."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_name, token=HF_TOKEN)
        total_bytes = 0
        for s in (getattr(info, "siblings", None) or []):
            size = getattr(s, "size", None)
            if size is not None:
                try:
                    total_bytes += int(size)
                except Exception:
                    pass
        return total_bytes / (1024 ** 3)
    except Exception:
        return 0.0

# --------------------
# Helpers
# --------------------
def build_test_sentence(tokenizer):
    mask = getattr(tokenizer, "mask_token", None)
    if not mask:
        for tok in ("[MASK]", "<mask>", "<MASK>"):
            try:
                if tok in tokenizer.get_vocab():
                    mask = tok
                    break
            except Exception:
                pass
    if not mask:
        return None
    return f"The capital of France is {mask}."

def try_fill_mask(model_name: str, cache_dir: Path):
    # --- size guard BEFORE any download  <— NEW
    size_gb = repo_size_gb(model_name)
    if size_gb and size_gb > MAX_REPO_SIZE_GB:
        return "SKIP_LARGE", f"Repo {size_gb:.2f} GB > {MAX_REPO_SIZE_GB:.2f} GB"

    # 1) load tokenizer/model (retry with trust_remote_code if needed)
    try:
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir), use_fast=True, token=HF_TOKEN)
        mdl = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=str(cache_dir), token=HF_TOKEN)
    except Exception as e:
        if "trust_remote_code" in str(e).lower():
            tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir), use_fast=True,
                                                trust_remote_code=True, token=HF_TOKEN)
            mdl = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=str(cache_dir),
                                                       trust_remote_code=True, token=HF_TOKEN)
        else:
            raise

    text = build_test_sentence(tok)
    if not text:
        return "FAIL", "No mask token (likely not a masked-LM)."

    # 2) run pipeline on GPU if available; on OOM, retry CPU
    device = 0 if torch.cuda.is_available() else -1
    try:
        nlp = pipeline("fill-mask", model=mdl, tokenizer=tok, device=device)
        out = nlp(text, top_k=TOP_K)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == 0:
            torch.cuda.empty_cache()
            nlp = pipeline("fill-mask", model=mdl, tokenizer=tok, device=-1)
            out = nlp(text, top_k=TOP_K)
        else:
            raise

    if isinstance(out, list) and len(out) > 0:
        return "OK", ""
    return "FAIL", "No predictions returned."

def clean_dir(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

# --------------------
# Main (sequential, one-by-one)
# --------------------
def mark_unsupported_once():
    # mark unsupported libs as SKIP so they won’t block
    for lib in UNSUPPORTED_LIBS:
        cur.execute("""
            SELECT model_id FROM Models
             WHERE problem = ? AND library = ?
               AND (health_status IS NULL OR health_status = '')
        """, (PROBLEM, lib))
        for (mid,) in cur.fetchall():
            update_health(mid, "SKIP", f"Library '{lib}' not supported by this Python runner")

def fetch_next():
    cur.execute("""
        SELECT model_id, model_name, downloads, library
          FROM Models
         WHERE problem = ?
           AND library = ?
           AND health_status IS NULL
         ORDER BY downloads DESC
         LIMIT 1
    """, (PROBLEM, SUPPORTED_LIB))
    return cur.fetchone()

def count_remaining():
    cur.execute("""
        SELECT COUNT(*) FROM Models
         WHERE problem = ?
           AND library = ?
           AND health_status IS NULL
    """, (PROBLEM, SUPPORTED_LIB))
    return cur.fetchone()[0]

def main():
    ensure_columns()
    mark_unsupported_once()

    remaining = count_remaining()
    print(f"Starting sequential health check for '{PROBLEM}' | remaining (transformers): {remaining}")

    i = 0
    while True:
        row = fetch_next()
        if not row:
            print("No more models to test.")
            break

        model_id, model_name, downloads, library = row
        i += 1
        print(f"[{i}/{remaining}] Testing: {model_name}  | downloads: {downloads:,}  | lib: {library}")

        model_cache = CACHE_ROOT / f"{model_id.replace('/', '_').replace('@', '_')}"
        model_cache.mkdir(parents=True, exist_ok=True)

        try:
            status, err = try_fill_mask(model_name, model_cache)
            update_health(model_id, status, err)
            print("   ->", "OK" if status == "OK" else f"{status}: {err[:120]}")
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
            print("   -> ERROR:", str(e)[:180])

        finally:
            clean_dir(model_cache)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nSummary (updated rows):")
    cur.execute("""
        SELECT health_status, COUNT(*)
          FROM Models
         WHERE problem = ?
           AND health_status IS NOT NULL
        GROUP BY health_status
        ORDER BY COUNT(*) DESC
    """, (PROBLEM,))
    for status, cnt in cur.fetchall():
        print(f"  {status}: {cnt}")

if __name__ == "__main__":
    try:
        main()
    finally:
        conn.close()
