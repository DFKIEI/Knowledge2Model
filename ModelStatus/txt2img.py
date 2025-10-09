# model_status_txt2img_min.py
import os, sqlite3, gc, shutil
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi
from diffusers import AutoPipelineForText2Image
import torch

HF_TOKEN = ""        # Add your token here if you want to access private models
DATABASE_PATH = "./huggingface2.db"
PROBLEM = "text-to-image"
LIBS_SUPPORTED = {"diffusers"}                # keep minimal
MIN_DOWNLOADS = 15                            
TEST_PROMPT = "a small 2D icon of a cat, simple, flat"
IMG_H, IMG_W = 256, 256                       # small to avoid OOM
MAX_REPO_SIZE_GB = 50.0                       # skip huge models


# Setup cache directory and fast downloads
TEMP_FOLDER = Path(__file__).resolve().parent / "model_testing_workspace"
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TEMP_FOLDER)

# Disable symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Enable Hugging Face fast downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
try:
    import hf_transfer
    print("✓ Fast downloads enabled with hf_transfer")
except ImportError:
    print("⚠ hf_transfer not installed. Install with: pip install hf-transfer")
    print("  Downloads will use standard speed.")


def ensure_health_columns(cur):
    cur.execute("PRAGMA table_info(Models)")
    cols = {c[1] for c in cur.fetchall()}
    if "health_status" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN health_status TEXT")
    if "health_error"  not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN health_error TEXT")
    if "last_checked"  not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN last_checked TIMESTAMP")

def update_health(cur, model_id, status, err=""):
    cur.execute("""
        UPDATE Models
        SET health_status=?, health_error=?, last_checked=?
        WHERE model_id=?
    """, (status, err[:500], datetime.now(timezone.utc).isoformat(), model_id))

def repo_is_too_big(model_name: str, max_gb: float) -> tuple[bool, float]:
    """Check if repository size exceeds limit by summing weight files."""
    try:
        api = HfApi()
        info = api.repo_info(
            repo_id=model_name,
            repo_type="model",
            files_metadata=True,
            token=HF_TOKEN or None
        )
        
        bytes_total = 0
        for f in info.siblings:
            name = getattr(f, "rfilename", "") or getattr(f, "path", "")
            size = getattr(f, "size", 0) or 0
            if name.endswith((".safetensors", ".bin", ".gguf", ".ckpt")):
                bytes_total += size
                
        size_gb = bytes_total / (1024 ** 3)
        return (size_gb >= max_gb, size_gb)
    except Exception:
        return (False, 0.0)

def cleanup_cache():
    """Clean up the temporary cache folder."""
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
        pass

def test_diffusers_model(model_name: str) -> tuple[str, str]:
    """Minimal 'does it run' check with auto-detection."""
    try:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load pipeline directly with auto-detection
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_name,
            cache_dir=str(TEMP_FOLDER),
            torch_dtype=dtype,
            token=HF_TOKEN or None,
            trust_remote_code=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        out = pipe(
            TEST_PROMPT,
            num_inference_steps=1,
            guidance_scale=0.0,
            height=IMG_H, width=IMG_W
        )

        ok = bool(getattr(out, "images", None)) and len(out.images) > 0
        del out, pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return ("OK", "") if ok else ("FAIL", "No image output")

    except Exception as e:
        msg = str(e)
        if "404" in msg or "not found" in msg.lower():
            return "NOT_FOUND", "Model not found"
        if "out of memory" in msg.lower() or "CUDA out of memory" in msg:
            return "OOM", "Out of memory"
        if "gated" in msg.lower() or "access" in msg.lower():
            return "GATED", "Gated repository"
        return "FAIL", msg[:300]

def main():
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    ensure_health_columns(cur); conn.commit()

    
    cur.execute(f"""
        SELECT model_id, model_name, downloads, library
        FROM Models
        WHERE problem = ?
          AND (health_status IS NULL OR health_status = '')
          AND library IN ({",".join("?"*len(LIBS_SUPPORTED))})
          AND downloads >= ?
        ORDER BY downloads DESC
    """, (PROBLEM, *LIBS_SUPPORTED, MIN_DOWNLOADS))
    rows = cur.fetchall()

    print(f"Testing {len(rows)} {PROBLEM} models (libs: {', '.join(LIBS_SUPPORTED)})")
    print(f"Cache directory: {TEMP_FOLDER}")
    print(f"Max repo size: {MAX_REPO_SIZE_GB} GB")

    for i, (model_id, model_name, downloads, lib) in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {model_name}  | {lib} | {downloads:,} dl")
        
        if lib not in LIBS_SUPPORTED:
            update_health(cur, model_id, "SKIP", f"Library {lib} not supported")
            conn.commit(); continue

        # Check repository size before downloading
        too_big, size_gb = repo_is_too_big(model_name, MAX_REPO_SIZE_GB)
        if too_big:
            update_health(cur, model_id, "SKIPPED_TOO_BIG", f"Repo≈{size_gb:.1f} GB")
            print(f"   -> SKIPPED_TOO_BIG (~{size_gb:.1f} GB)")
            conn.commit(); continue

        status, err = test_diffusers_model(model_name)
        update_health(cur, model_id, status, err)
        conn.commit()
        print(f"   -> {status}" + (f" ({err[:80]}...)" if err else ""))
        
        # Clean cache after each model to save space
        cleanup_cache()

    conn.close()
    
    # Final cleanup
    try:
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
    except:
        pass
        
    print("Done.")

if __name__ == "__main__":
    main()