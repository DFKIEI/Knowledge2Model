# model_status_feat_extract.py
import os
import gc
import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoFeatureExtractor, AutoProcessor


import warnings
warnings.filterwarnings("ignore")


HF_TOKEN = ""  # Add your token here if you want to access private models
DATABASE_PATH = "./huggingface2.db"
PROBLEM = "feature-extraction"
LIBS_TO_TEST = ("transformers", "sentence-transformers")
MIN_DOWNLOADS = 15
TEST_TEXTS = [
    "This is a test sentence for feature extraction.",
    "Machine learning models can generate embeddings.",
]
MAX_REPO_GB = 60.0 


ROOT = Path(__file__).resolve().parent
TEMP_FOLDER = ROOT / "feat_extract_workspace"
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TEMP_FOLDER)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

STATUSES = ["OK","FAIL","OOM","NOT_FOUND","GATED","TRUST_REQUIRED","SKIP","ERROR","SKIPPED_TOO_BIG"]


api = HfApi()    


def ensure_health_columns(cur):
    cur.execute("PRAGMA table_info(Models)")
    cols = {c[1] for c in cur.fetchall()}
    if "health_status" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN health_status TEXT")
    if "health_error" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN health_error TEXT")
    if "last_checked" not in cols:
        cur.execute("ALTER TABLE Models ADD COLUMN last_checked TIMESTAMP")


def update_health(conn, cur, model_id, status, error=""):
    cur.execute(
        """
        UPDATE Models
           SET health_status = ?,
               health_error  = ?,
               last_checked  = ?
         WHERE model_id = ?
        """,
        (status, (error or "")[:500], datetime.now(timezone.utc).isoformat(), model_id),
    )
    conn.commit()

def cleanup_cache():
    """Delete all contents of TEMP_FOLDER (keep the folder)."""
    if not TEMP_FOLDER.exists():
        return
    for item in list(TEMP_FOLDER.iterdir()):
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        except Exception:
            pass


def repo_is_too_big(model_name: str, max_gb: float) -> tuple[bool, float]:
    """Estimate total size of weight files in the repo without downloading them."""
    try:
        info = api.repo_info(
            repo_id=model_name,
            repo_type="model",
            files_metadata=True,
            token=HF_TOKEN or None,
        )
        total = 0
        for f in info.siblings:
            name = getattr(f, "rfilename", "") or getattr(f, "path", "")
            size = getattr(f, "size", 0) or 0
            if name.endswith((".safetensors", ".bin", ".gguf", ".pt")):
                total += size
        return (total / (1024**3) >= max_gb, total / (1024**3))
    except Exception:
        
        return (False, 0.0)


def classify_error(msg: str):
    m = (msg or "").lower()
    if "404" in m or "not found" in m:
        return "NOT_FOUND", "Model not found"
    if "out of memory" in m or "cuda out of memory" in m or "oom" in m:
        return "OOM", "Out of memory"
    if any(x in m for x in ["gated", "forbidden", "access", "private", "401", "403"]):
        return "GATED", "Gated repository - needs access"
    if "trust_remote_code" in m:
        return "TRUST_REQUIRED", "Requires trust_remote_code"
    return "FAIL", (msg or "")[:500]



def tiny_image():
    """Deterministic tiny RGB image for quick vision probe."""
    h, w = 96, 96
    arr = (np.arange(h * w * 3) % 255).astype(np.uint8).reshape(h, w, 3)
    from PIL import Image
    return Image.fromarray(arr, "RGB")

def tiny_wave(sr: int = 16000, seconds: float = 1.0) -> tuple[np.ndarray, int]:
    """Deterministic 440 Hz sine wave at ~-20 dBFS for 1 second."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    x = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    return x, sr



def test_transformers_model(model_name: str):
    """
    Minimal, config-aware probe (TEXT / IMAGE / AUDIO; first success wins).
    - Uses AutoConfig to pick a smart order
    - One-shot trust_remote_code retry for loaders
    - Mean-pools text if no pooler_output
    - Respects tokenizer.model_max_length (<=512)
    - Normalizes audio sampling_rate + input_values
    - If fp16 fails on CUDA, retries forward once in fp32
    """
    from transformers import AutoConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefer_fp16 = device == "cuda"
    dtype = torch.float16 if prefer_fp16 else torch.float32
    errors = []


    def load_retry(loader, **kw):
        try:
            return loader(trust_remote_code=False, **kw)
        except Exception as e:
            if "trust_remote_code" in str(e).lower():
                return loader(trust_remote_code=True, **kw)
            raise

    def mean_pool(last_hidden_state, mask):
        mask = mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def ok(feats, expected_batch=None):
        if feats is None or not hasattr(feats, "numel"): return False, "no tensor"
        if feats.numel() == 0 or not torch.isfinite(feats).all(): return False, "non-finite/empty"
        if expected_batch is not None and hasattr(feats, "shape") and feats.shape[0] != expected_batch:
            return False, f"unexpected batch {tuple(feats.shape)}"
        return True, ""

    def load_model():
        m = load_retry(
            AutoModel.from_pretrained,
            pretrained_model_name_or_path=model_name,
            cache_dir=str(TEMP_FOLDER),
            use_auth_token=HF_TOKEN or None,
            low_cpu_mem_usage=True
        )
        return m.to(device=device, dtype=dtype)

    def try_forward(fwd):
        nonlocal dtype
        try:
            return fwd()
        except Exception as e:
            if prefer_fp16 and dtype == torch.float16:
                dtype = torch.float32
                return fwd()
            raise e

    def get_image_processor():
        try:
            return load_retry(
                AutoImageProcessor.from_pretrained,
                pretrained_model_name_or_path=model_name,
                cache_dir=str(TEMP_FOLDER),
                use_auth_token=HF_TOKEN or None
            )
        except Exception:
            return load_retry(
                AutoFeatureExtractor.from_pretrained,
                pretrained_model_name_or_path=model_name,
                cache_dir=str(TEMP_FOLDER),
                use_auth_token=HF_TOKEN or None
            )

    def get_audio_inputs():
        try:
            proc = load_retry(
                AutoProcessor.from_pretrained,
                pretrained_model_name_or_path=model_name,
                cache_dir=str(TEMP_FOLDER),
                use_auth_token=HF_TOKEN or None
            )
            sr = (getattr(getattr(proc, "feature_extractor", None), "sampling_rate", None)
                  or getattr(proc, "sampling_rate", None) or 16000)
            wave, _ = tiny_wave(sr)
            inputs = proc(wave, sampling_rate=sr, return_tensors="pt")
        except Exception:
            fe = load_retry(
                AutoFeatureExtractor.from_pretrained,
                pretrained_model_name_or_path=model_name,
                cache_dir=str(TEMP_FOLDER),
                use_auth_token=HF_TOKEN or None
            )
            sr = getattr(fe, "sampling_rate", None) or 16000
            wave, _ = tiny_wave(sr)
            inputs = fe(wave, sampling_rate=sr, return_tensors="pt")
        if "input_values" not in inputs and "inputs" in inputs:
            inputs["input_values"] = inputs.pop("inputs")
        return {k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device)) for k, v in inputs.items()}

    try:
        cfg = load_retry(
            AutoConfig.from_pretrained,
            pretrained_model_name_or_path=model_name,
            use_auth_token=HF_TOKEN or None
        )
        mtype = (getattr(cfg, "model_type", "") or "").lower()
        archs = [a.lower() for a in (getattr(cfg, "architectures", []) or [])]
        has = lambda tag: (tag in mtype) or any(tag in a for a in archs)
        is_seq2seq = any(has(t) for t in ("t5","bart","mbart","whisper"))
        is_audio   = any(has(t) for t in ("wav2vec","hubert","whisper","clap","encodec","mimi"))
        is_vision  = any(has(t) for t in ("vit","beit","swin","convnext","dinov2","siglip_vision","clipvision","clip"))
    except Exception:
        is_seq2seq = is_audio = is_vision = False


    def probe_text():
        tok = load_retry(
            AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path=model_name,
            cache_dir=str(TEMP_FOLDER),
            use_auth_token=HF_TOKEN or None,
            use_fast=True
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        max_len = min(int(getattr(tok, "model_max_length", 512) or 512), 512)
        model = load_model()
        if is_seq2seq and hasattr(model, "get_encoder"):
            model = model.get_encoder().to(device=device, dtype=dtype)
        batch = tok(TEST_TEXTS, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}

        def fwd():
            nonlocal model
            model = model.to(device=device, dtype=dtype)
            with torch.inference_mode():
                out = model(**batch)
            last = getattr(out, "last_hidden_state", None)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None and last is not None:
                pooled = mean_pool(last, batch.get("attention_mask", torch.ones(last.size()[:2], device=device)))
            feats = pooled if pooled is not None else last
            good, why = ok(feats, expected_batch=len(TEST_TEXTS))
            return ("OK", f"Text {tuple(feats.shape)}") if good else (None, f"text: {why}")
        return try_forward(fwd)

    def probe_image():
        proc = get_image_processor()
        img = tiny_image()
        model = load_model()
        inputs = proc(img, return_tensors="pt"); inputs = {k: v.to(device) for k, v in inputs.items()}
        def fwd():
            nonlocal model
            model = model.to(device=device, dtype=dtype)
            with torch.inference_mode():
                out = model(**inputs)
            feats = getattr(out, "last_hidden_state", None) or getattr(out, "pooler_output", None)
            good, why = ok(feats)
            return ("OK", f"Image {tuple(feats.shape)}") if good else (None, f"image: {why}")
        return try_forward(fwd)

    def probe_audio():
        inputs = get_audio_inputs()
        model = load_model()
        def fwd():
            nonlocal model
            model = model.to(device=device, dtype=dtype)
            with torch.inference_mode():
                out = model(**inputs)
            feats = (getattr(out, "last_hidden_state", None)
                     or getattr(out, "extract_features", None)
                     or getattr(out, "pooler_output", None))
            good, why = ok(feats)
            return ("OK", f"Audio {tuple(feats.shape)}") if good else (None, f"audio: {why}")
        return try_forward(fwd)

    
    ordered = []
    if is_audio:  ordered.append(probe_audio)
    if is_vision: ordered.append(probe_image)
    ordered.append(probe_text)
    for p in (probe_image, probe_audio, probe_text):
        if p not in ordered: ordered.append(p)

    for p in ordered:
        try:
            res = p()
            if isinstance(res, tuple) and res[0] == "OK":
                return res
            if isinstance(res, tuple) and res[0] is None:
                errors.append(res[1])
        except Exception as e:
            errors.append(str(e))

    combined = " | ".join(err[:140] for err in errors if err)
    return classify_error(combined)


def test_sentence_transformers_model(model_name: str):
    """Text-only probe (image is handled in transformers path); falls back to transformers if ST itself fails."""
    try:
        from sentence_transformers import SentenceTransformer

        def load_st(trust: bool = False):
            return SentenceTransformer(
                model_name,
                cache_folder=str(TEMP_FOLDER),
                use_auth_token=HF_TOKEN or None,
                trust_remote_code=trust,  # <-- added
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        
        try:
            model = load_st(trust=False)
        except Exception as e:
            if "trust_remote_code" in str(e).lower():
                model = load_st(trust=True)  
            else:
                raise

        emb = model.encode(
            TEST_TEXTS,
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=min(8, len(TEST_TEXTS)),
            show_progress_bar=False,
        )

        if not isinstance(emb, np.ndarray):
            emb = np.asarray(emb)

        if emb.ndim != 2 or emb.shape[0] != len(TEST_TEXTS) or emb.shape[1] <= 0:
            return "FAIL", f"Invalid embedding shape: {getattr(emb, 'shape', None)}"
        if not np.isfinite(emb).all():
            return "FAIL", "Non-finite values in embeddings"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return "OK", f"Embeddings {emb.shape}"

    except Exception as e:
        msg = str(e)[:500]
        if "sentence-transformers" in msg.lower() or "sbert" in msg.lower():
            return test_transformers_model(model_name)
        return classify_error(msg)



def main():
    print("=" * 60)
    print("FEATURE EXTRACTION MODEL HEALTH CHECK")
    print("=" * 60)
    print(f"Libraries: {', '.join(LIBS_TO_TEST)} | Min downloads: {MIN_DOWNLOADS}")
    print(f"Cache dir: {TEMP_FOLDER}")
    print("✓ Using HuggingFace token" if HF_TOKEN and HF_TOKEN != "hf_YOUR_TOKEN_HERE" else "⚠ No HF token (private/gated may fail)")
    print("=" * 60 + "\n")

    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    ensure_health_columns(cur)
    conn.commit()

    placeholders = ",".join("?" for _ in LIBS_TO_TEST)
    cur.execute(
        f"""
        SELECT model_id, model_name, downloads, library
          FROM Models
         WHERE problem = ?
           AND library IN ({placeholders})
           AND (health_status IS NULL OR health_status = '')
           AND downloads >= ?
         ORDER BY downloads DESC
        """,
        (PROBLEM, *LIBS_TO_TEST, MIN_DOWNLOADS),
    )
    models = cur.fetchall()
    total = len(models)
    print(f"Found {total} models to test\n")

    stats = {k: 0 for k in STATUSES}

    for i, (model_id, model_name, downloads, library) in enumerate(models, 1):
        print(f"[{i}/{total}] {model_name}  |  lib: {library}  |  dls: {downloads:,}")

        too_big, size_gb = repo_is_too_big(model_name, MAX_REPO_GB)
        if too_big:
            update_health(conn, cur, model_id, "SKIPPED_TOO_BIG", f"Repo≈{size_gb:.1f} GB")
            stats["SKIPPED_TOO_BIG"] = stats.get("SKIPPED_TOO_BIG", 0) + 1
            print(f"   → SKIPPED_TOO_BIG (~{size_gb:.1f} GB)")
            cleanup_cache()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        try:
            if library == "transformers":
                status, err = test_transformers_model(model_name)
            elif library == "sentence-transformers":
                status, err = test_sentence_transformers_model(model_name)
            else:
                status, err = "SKIP", f"Library {library} not implemented"

            update_health(conn, cur, model_id, status, err)
            stats[status] = stats.get(status, 0) + 1
            print(f"   → {status}: {err[:120] if err else ''}")
        except Exception as e:
            s, m = classify_error(str(e))
            update_health(conn, cur, model_id, s, m)
            stats[s] = stats.get(s, 0) + 1
            print(f"   → {s}: {m[:120]}")
        finally:
            cleanup_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    done = sum(stats.values())
    for k in STATUSES:
        if stats.get(k, 0):
            pct = (100.0 * stats[k] / done) if done else 0.0
            print(f"  {k}: {stats[k]} ({pct:.1f}%)")
    print(f"\n  TOTAL TESTED: {done}")

    print("\nBreakdown by library:")
    for lib in LIBS_TO_TEST:
        cur.execute(
            """
            SELECT health_status, COUNT(*)
              FROM Models
             WHERE problem = ? AND library = ? AND health_status IS NOT NULL
             GROUP BY health_status
            """,
            (PROBLEM, lib),
        )
        rows = cur.fetchall()
        if rows:
            print(f"\n  {lib}:")
            for s, c in rows:
                print(f"    {s}: {c}")

    conn.close()

    try:
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
    except Exception:
        pass

    print("\n✓ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        cleanup_cache()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        cleanup_cache()
