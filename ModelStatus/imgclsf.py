"""
Image Classification Model Health Checker
Tests image-classification models from HuggingFace database.
Supports: transformers, timm libraries
"""

import os
import sqlite3
import shutil
import warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================

HF_TOKEN = ""  # Your HuggingFace token (or set HF_TOKEN env var)
DATABASE_PATH = "" # Path to your database

PROBLEM = "image-classification"
LIBRARIES = ["transformers", "timm"]

# Filtering thresholds
MIN_DOWNLOADS = 100        # Only test models with at least this many downloads
MAX_REPO_SIZE_GB = 60.0    # Skip repos larger than this (GB)

# ============================================================================
# SETUP
# ============================================================================

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "model_testing_workspace_imgcls"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_ROOT)

# Get token from env if not set above
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN", "")

# ============================================================================
# DATABASE
# ============================================================================

conn = sqlite3.connect(DATABASE_PATH)
cur = conn.cursor()


def ensure_columns():
    """Add health tracking columns if they don't exist."""
    cur.execute("PRAGMA table_info(Models)")
    cols = {c[1] for c in cur.fetchall()}
    
    if "health_status" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN health_status TEXT")
    if "health_error" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN health_error TEXT")
    if "last_checked" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN last_checked TIMESTAMP")
    conn.commit()


def update_health(model_id, status, error=""):
    """Update model health status in database."""
    cur.execute("""
        UPDATE Models 
        SET health_status=?, health_error=?, last_checked=?
        WHERE model_id=?
    """, (status, (error or "")[:500], datetime.now(timezone.utc).isoformat(), model_id))
    conn.commit()


def fetch_next_model(library):
    """Fetch next untested model for a library."""
    cur.execute("""
        SELECT model_id, model_name, downloads, library
        FROM Models
        WHERE problem=? AND library=? AND health_status IS NULL AND downloads >= ?
        ORDER BY downloads DESC
        LIMIT 1
    """, (PROBLEM, library, MIN_DOWNLOADS))
    return cur.fetchone()


def count_remaining(library):
    """Count remaining untested models."""
    cur.execute("""
        SELECT COUNT(*) FROM Models
        WHERE problem=? AND library=? AND health_status IS NULL AND downloads >= ?
    """, (PROBLEM, library, MIN_DOWNLOADS))
    return cur.fetchone()[0]


# ============================================================================
# TEST IMAGES
# ============================================================================

def create_test_images():
    """Create PIL test images for model inference."""
    from PIL import Image
    import numpy as np
    
    h, w = 224, 224
    images = []
    
    # Gradient image
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    gradient = np.stack([xv, yv, ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    images.append(Image.fromarray(gradient, mode='RGB'))
    
    # Solid color image
    solid = np.zeros((h, w, 3), dtype=np.uint8)
    solid[..., 1] = 180
    images.append(Image.fromarray(solid, mode='RGB'))
    
    return images


# ============================================================================
# REPO SIZE CHECK
# ============================================================================

def get_repo_size_gb(model_name):
    """Get HuggingFace repo size in GB."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_name, token=HF_TOKEN or None)
        total = sum(int(s.size) for s in (info.siblings or []) if getattr(s, 'size', None))
        return total / (1024 ** 3)
    except:
        return 0.0


# ============================================================================
# TRANSFORMERS TESTER
# ============================================================================

def test_transformers(model_name, test_images):
    """Test a transformers image-classification model."""
    import torch
    from transformers import pipeline
    
    # Check repo size
    size_gb = get_repo_size_gb(model_name)
    if size_gb > MAX_REPO_SIZE_GB:
        return "SKIP_LARGE", f"Repo {size_gb:.1f}GB > {MAX_REPO_SIZE_GB}GB limit"
    
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        pipe = pipeline(
            "image-classification",
            model=model_name,
            device=device,
            token=HF_TOKEN or None,
            trust_remote_code=True,
            batch_size=1
        )
        
        for img in test_images:
            out = pipe(img)
            if isinstance(out, list) and out and "label" in out[0] and "score" in out[0]:
                return "OK", ""
        
        return "FAIL", "No valid predictions"
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == 0:
            torch.cuda.empty_cache()
            # Retry on CPU
            try:
                pipe = pipeline("image-classification", model=model_name, device=-1, 
                               token=HF_TOKEN or None, trust_remote_code=True, batch_size=1)
                out = pipe(test_images[0])
                if isinstance(out, list) and out and "label" in out[0]:
                    return "OK", ""
                return "FAIL", "CPU retry: no valid output"
            except Exception as cpu_e:
                return "OOM", f"GPU OOM, CPU failed: {str(cpu_e)[:100]}"
        raise


# ============================================================================
# TIMM TESTER
# ============================================================================

def get_timm_name(hf_name):
    """Convert HuggingFace model name to TIMM format."""
    import timm
    
    available = set(timm.list_models(pretrained=True))
    
    # Try exact match
    if hf_name in available:
        return hf_name
    
    # Extract base name and try variations
    base = hf_name.split("/")[-1]
    
    if base in available:
        return base
    
    # Convert to TIMM style (hyphens to underscores)
    timm_style = base.replace("-", "_").lower()
    if timm_style in available:
        return timm_style
    
    # Try without underscores
    no_underscore = timm_style.replace("_", "")
    if no_underscore in available:
        return no_underscore
    
    return None


def test_timm(model_name, test_images):
    """Test a TIMM image classification model."""
    import torch
    import timm
    from torchvision import transforms
    
    # Map to TIMM name
    timm_name = get_timm_name(model_name)
    if not timm_name:
        return "SKIP", f"Not found in TIMM registry"
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = timm.create_model(timm_name, pretrained=True).eval().to(device)
        x = preprocess(test_images[0]).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            out = model(x)
        
        if hasattr(out, 'shape') and len(out.shape) == 2 and out.shape[1] > 1 and torch.isfinite(out).all():
            return "OK", ""
        
        return "FAIL", f"Invalid output shape: {tuple(out.shape)}"
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            torch.cuda.empty_cache()
            # Retry on CPU
            try:
                model = timm.create_model(timm_name, pretrained=True).eval()
                x = preprocess(test_images[0]).unsqueeze(0)
                with torch.inference_mode():
                    out = model(x)
                if hasattr(out, 'shape') and len(out.shape) == 2 and torch.isfinite(out).all():
                    return "OK", ""
                return "FAIL", "CPU retry: invalid output"
            except Exception as cpu_e:
                return "OOM", f"GPU OOM, CPU failed: {str(cpu_e)[:100]}"
        raise


# ============================================================================
# UTILITIES
# ============================================================================

def clean_dir(path):
    """Remove directory safely."""
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except:
        pass


def clear_gpu():
    """Clear GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass


def categorize_error(error_msg):
    """Categorize an exception into a status code."""
    msg = str(error_msg).lower()
    
    if "404" in msg or "not found" in msg:
        return "NOT_FOUND", "Model not found"
    elif "out of memory" in msg:
        return "OOM", "Out of memory"
    elif "trust_remote_code" in msg:
        return "TRUST_NEEDED", "Requires trust_remote_code"
    elif "is not a supported task" in msg:
        return "FAIL", "Task not supported"
    else:
        return "FAIL", str(error_msg)[:300]


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("IMAGE CLASSIFICATION MODEL HEALTH CHECKER")
    print("=" * 60)
    print(f"Database:      {DATABASE_PATH}")
    print(f"Min Downloads: {MIN_DOWNLOADS:,}")
    print(f"Max Repo Size: {MAX_REPO_SIZE_GB} GB")
    print(f"Libraries:     {LIBRARIES}")
    print(f"HF Token:      {'Set' if HF_TOKEN else 'Not set'}")
    print()
    
    ensure_columns()
    
    # Create test images
    print("Creating test images...")
    test_images = create_test_images()
    
    # Test each library
    for library in LIBRARIES:
        remaining = count_remaining(library)
        print(f"\n{'=' * 60}")
        print(f"{library.upper()} | Remaining: {remaining} | Min Downloads: {MIN_DOWNLOADS:,}")
        print("=" * 60)
        
        if remaining == 0:
            print(f"No {library} models to test")
            continue
        
        tested = 0
        stats = {"OK": 0, "FAIL": 0, "OOM": 0, "NOT_FOUND": 0, "SKIP": 0, "SKIP_LARGE": 0}
        
        while True:
            row = fetch_next_model(library)
            if not row:
                break
            
            model_id, model_name, downloads, lib = row
            tested += 1
            
            print(f"\n[{library} #{tested}] {model_name}")
            print(f"           Downloads: {downloads:,}")
            
            # Cache directory for this model
            safe_name = model_id.replace("/", "_").replace("@", "_").replace(":", "_")
            model_cache = CACHE_ROOT / f"{library}_{safe_name}"
            model_cache.mkdir(parents=True, exist_ok=True)
            
            try:
                if library == "transformers":
                    status, error = test_transformers(model_name, test_images)
                elif library == "timm":
                    status, error = test_timm(model_name, test_images)
                else:
                    status, error = "SKIP", f"Unsupported library: {library}"
                
                update_health(model_id, status, error)
                stats[status] = stats.get(status, 0) + 1
                
                if status == "OK":
                    print("           -> OK")
                else:
                    print(f"           -> {status}: {error[:80]}")
                    
            except Exception as e:
                status, error = categorize_error(e)
                update_health(model_id, status, error)
                stats[status] = stats.get(status, 0) + 1
                print(f"           -> ERROR: {error[:80]}")
                
            finally:
                clean_dir(model_cache)
                clear_gpu()
        
        # Library summary
        print(f"\n{library.upper()} Summary:")
        for s, c in stats.items():
            if c > 0:
                print(f"  {s}: {c}")
        print(f"  TOTAL: {tested}")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    cur.execute("""
        SELECT health_status, COUNT(*)
        FROM Models WHERE problem=? AND health_status IS NOT NULL
        GROUP BY health_status ORDER BY COUNT(*) DESC
    """, (PROBLEM,))
    
    total = 0
    for status, count in cur.fetchall():
        print(f"  {status}: {count}")
        total += count
    print(f"  TOTAL TESTED: {total}")


def cleanup():
    """Cleanup resources."""
    try:
        conn.close()
    except:
        pass
    clean_dir(CACHE_ROOT)


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