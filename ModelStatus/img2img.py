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
PROBLEM = "image-to-image"
SUPPORTED_LIB = "diffusers"           # only test diffusers; others remain NULL
MAX_REPO_SIZE_GB = 60.0               # hard cap

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "model_testing_workspace_img2img"
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

# ===== Tiny synthetic init image =====
def make_init_image(size=512):
    """Return a simple 512x512 RGB gradient PIL image for img2img."""
    import numpy as np
    from PIL import Image
    w = h = size
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ys = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(xs, ys)
    img = np.stack([xv, yv, ((xv // 2) + (yv // 2)).astype(np.uint8)], axis=-1)
    return Image.fromarray(img, mode="RGB")

TEST_PROMPT = "A watercolor painting of a small cozy cabin."

# ===== Single-model test (diffusers) =====
def try_diffusers_img2img(model_name: str):
    # Size gate BEFORE any download
    size_gb = repo_size_gb(model_name)
    if size_gb and size_gb > MAX_REPO_SIZE_GB:
        return "SKIP_LARGE", f"Repo {size_gb:.2f} GB > {MAX_REPO_SIZE_GB:.2f} GB"

    import torch
    from diffusers import AutoPipelineForImage2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    init_image = make_init_image(512)

    # Load pipeline
    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            token=HF_TOKEN
        )
    except Exception as e:
        msg = str(e).lower()
        if "403" in msg or "not authorized" in msg or "private" in msg or "forbidden" in msg:
            return "ACCESS_DENIED", "Model access denied/private"
        if "not found" in msg or "404" in msg:
            return "NOT_FOUND", "Model or files not found"
        # Not an image-to-image pipeline or other issue
        return "FAIL", str(e)[:300]

    # Move to device
    try:
        pipe = pipe.to(device)
    except Exception:
        pass

    # Inference (very light settings)
    gen = torch.Generator(device=device).manual_seed(0)
    try:
        image = pipe(
            prompt=TEST_PROMPT,
            image=init_image,
            num_inference_steps=5,
            guidance_scale=5.0,
            strength=0.7,
            generator=gen
        ).images[0]
    except RuntimeError as e:
        # Retry on CPU if CUDA OOM
        if "out of memory" in str(e).lower() and device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            pipe = pipe.to("cpu")
            gen = torch.Generator(device="cpu").manual_seed(0)
            image = pipe(
                prompt=TEST_PROMPT,
                image=init_image,
                num_inference_steps=5,
                guidance_scale=5.0,
                strength=0.7,
                generator=gen
            ).images[0]
        else:
            return "FAIL", str(e)[:300]
    except Exception as e:
        return "FAIL", str(e)[:300]

    # Pass if we got a PIL.Image back
    try:
        from PIL.Image import Image as PILImage
        if isinstance(image, PILImage):
            return "OK", ""
    except Exception:
        pass
    return "FAIL", "Did not receive an output image."

# ===== FS utils =====
def clean_dir(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

# ===== Sequential fetch-next (diffusers only) =====
def fetch_next():
    cur.execute("""
        SELECT model_id, model_name, downloads, library
          FROM Models
         WHERE problem=? AND library=?
           AND health_status IS NULL
         ORDER BY downloads DESC
         LIMIT 1
    """, (PROBLEM, SUPPORTED_LIB))
    return cur.fetchone()

def count_remaining():
    cur.execute("""
        SELECT COUNT(*) FROM Models
         WHERE problem=? AND library=?
           AND health_status IS NULL
    """, (PROBLEM, SUPPORTED_LIB))
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
            status, err = try_diffusers_img2img(model_name)
            update_health(model_id, status, err)
            print("   ->", "OK" if status == "OK" else f"{status}: {err[:160]}")
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg:
                update_health(model_id, "NOT_FOUND", str(e))
            elif "out of memory" in msg:
                update_health(model_id, "OOM", "Out of memory")
            elif "not authorized" in msg or "forbidden" in msg or "private" in msg:
                update_health(model_id, "ACCESS_DENIED", "Model access denied/private")
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
