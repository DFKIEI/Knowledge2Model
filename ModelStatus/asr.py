import os, sqlite3, shutil, warnings, torch, numpy as np
from pathlib import Path
from datetime import datetime, timezone
from transformers import pipeline
from huggingface_hub import HfApi
import soundfile as sf  # pip install soundfile
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoModelForCTC, AutoProcessor, pipeline
warnings.filterwarnings("ignore")

HF_TOKEN = os.getenv("HF_TOKEN", "hf_xxx")           # set env or hardcode
DB_PATH  = os.getenv("HF_DB_PATH", "./Hugging2KG/huggingface2.db")
ROOT = Path(__file__).resolve().parent           # /path/to/model_status
HF_HOME = ROOT / "HF_WORKSPACE"                  # base cache
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)

# simple config
PROBLEM = "automatic-speech-recognition"
LIBRARY = "transformers"
MIN_DOWNLOADS = 15
ASR_TEST_WAV = os.getenv("ASR_TEST_WAV", "")         # optional spoken sample wav
MAX_REPO_SIZE_GB = float(os.getenv("MAX_REPO_SIZE_GB", "60"))

hf_api = HfApi()

# --- DB setup -------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(Models)")
cols = {c[1] for c in cursor.fetchall()}
if "health_status" not in cols: cursor.execute("ALTER TABLE Models ADD COLUMN health_status TEXT")
if "health_error"  not in cols: cursor.execute("ALTER TABLE Models ADD COLUMN health_error  TEXT")
if "last_checked"  not in cols: cursor.execute("ALTER TABLE Models ADD COLUMN last_checked  TIMESTAMP")
conn.commit()

def update_health(model_id, status, err=""):
    cursor.execute("""
        UPDATE Models
        SET health_status=?, health_error=?, last_checked=?
        WHERE model_id=?""",
        (status, err[:500], datetime.now(timezone.utc).isoformat(), model_id))
    conn.commit()

def get_next_model():
    """Fetch exactly one next untested model, highest downloads first."""
    cursor.execute("""
        SELECT model_id, model_name, downloads
        FROM Models
        WHERE health_status IS NULL
          AND problem = ?
          AND library = ?
          AND downloads >= ?
        ORDER BY downloads DESC
        LIMIT 1
    """, (PROBLEM, LIBRARY, MIN_DOWNLOADS))
    return cursor.fetchone()

# --- Repo size check -----------------------------------------------
def get_repo_size_gb(repo_id: str) -> float | None:
    """
    Returns total repo size in GB using HF metadata.
    If size cannot be determined, returns None (we won't block download).
    """
    try:
        # newer API: repo_info(..., files_metadata=True) exposes sizes
        info = hf_api.repo_info(
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
            files_metadata=True
        )
        # info.siblings is a list of files with .size (bytes)
        total_bytes = 0
        for f in getattr(info, "siblings", []) or []:
            sz = getattr(f, "size", None)
            if isinstance(sz, int):
                total_bytes += sz
        # some hubs return 0 if unknown
        if total_bytes > 0:
            return total_bytes / (1024**3)
        # fallback to legacy model_info if needed
        mi = hf_api.model_info(repo_id, token=HF_TOKEN)
        total_bytes = 0
        for f in getattr(mi, "siblings", []) or []:
            sz = getattr(f, "size", None)
            if isinstance(sz, int):
                total_bytes += sz
        return (total_bytes / (1024**3)) if total_bytes > 0 else None
    except Exception:
        return None  # on gated/private errors etc., don't block; we just can't pre-size

# --- Test audio -----------------------------------------------------
def load_test_audio():
    """Loads user-specified WAV or generates a quiet synthetic tone."""
    if ASR_TEST_WAV and Path(ASR_TEST_WAV).exists():
        audio, sr = sf.read(ASR_TEST_WAV, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # convert stereo to mono
    else:
        # generate a quiet 1-second 440 Hz tone (16 kHz sample rate)
        sr = 16000
        t = np.linspace(0, 1.0, int(sr), endpoint=False, dtype=np.float32)
        audio = 0.01 * np.sin(2 * np.pi * 440 * t, dtype=np.float32)

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    print("Audio shape:", audio.shape, "dtype:", audio.dtype, "sample rate:", sr)

    # return both for flexibility
    return {
        "raw": audio,
        "array": audio,
        "sampling_rate": int(sr)
    }


TEST_AUDIO = load_test_audio()

# --- Error mapping --------------------------------------------------
def map_error(e: Exception):
    msg, low = str(e), str(e).lower()
    if "out of memory" in low or ("cuda" in low and "oom" in low): return "OOM", "Out of memory"
    if "404" in msg or "not found" in low or "could not find" in low: return "NOT_FOUND", "Model not found"
    if "trust_remote_code" in low: return "TRUST_NEEDED", "Requires trust_remote_code"
    return "FAIL", msg[:200]

# --- Single-model test ----------------------------------------------

def test_model_transformers(model_name, cache_dir):

    for trust in (False, True):  # retry once with trust=True if needed
        try:
            # --- Load config to detect model family ---
            config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                token=HF_TOKEN,
                trust_remote_code=trust
            )

            # --- Detect model type ---
            cfg_cls = config.__class__.__name__
            if "SpeechEncoderDecoder" in cfg_cls or "Whisper" in cfg_cls:
                family = "seq2seq"
                ModelClass = AutoModelForSpeechSeq2Seq
            elif "Wav2Vec2" in cfg_cls or "Hubert" in cfg_cls or "MMS" in cfg_cls:
                family = "ctc"
                ModelClass = AutoModelForCTC
            else:
                family = "unknown"
                ModelClass = AutoModelForCTC  # safe fallback

            print(f"   Detected family: {family} ({cfg_cls})")

            # --- Load model + processor ---
            model = ModelClass.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                token=HF_TOKEN,
                trust_remote_code=trust
            )
            processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                token=HF_TOKEN,
                trust_remote_code=trust
            )

            # --- Create ASR pipeline ---
            asr = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
                feature_extractor=processor.feature_extractor if hasattr(processor, "feature_extractor") else processor,
                device=0 if torch.cuda.is_available() else -1,
            )

            # --- Run inference: try 'raw', fallback to 'array' ---
            try:
                out = asr(inputs={"raw": TEST_AUDIO["raw"], "sampling_rate": TEST_AUDIO["sampling_rate"]})
            except Exception:
                out = asr(inputs={"array": TEST_AUDIO["array"], "sampling_rate": TEST_AUDIO["sampling_rate"]})

            # --- Parse output ---
            text = out.get("text", "") if isinstance(out, dict) else (out or "")


            # --- Interpret results ---
            if isinstance(text, str):
                return "OK", "" if text.strip() else "Empty transcript (synthetic audio)"
            return "FAIL", "No text output"

        except Exception as e:
            status, err = map_error(e)
            if status == "TRUST_NEEDED" and not trust:
                continue  # retry with trust=True
            return status, err

    return "FAIL", "Unknown failure"



# --- Loop: one-by-one -----------------------------------------------
def main():
    print("="*60)
    print("MODEL HEALTH CHECK - ASR (single model loop, repo size guard)")
    print("="*60)
    print("✓ Using HF token" if HF_TOKEN else "⚠ No HF token (private models will fail)")
    print(f"Audio source: {'file' if ASR_TEST_WAV else 'synthetic tone'}")
    print(f"Max repo size: {MAX_REPO_SIZE_GB:.1f} GB")

    # optional: faster HF downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    tested = 0
    cursor.execute("""
        SELECT COUNT(*)
        FROM Models
        WHERE health_status IS NULL
          AND problem = ?
          AND library = ?
          AND downloads >= ?
    """, (PROBLEM, LIBRARY, MIN_DOWNLOADS))
    total_to_test = cursor.fetchone()[0]

    while True:
        row = get_next_model()
        if not row:
            print("\nNo more untested models matching the filter.")
            break

        model_id, model_name, downloads = row
        print(f"\n[{tested+1}/{total_to_test}] Testing: {model_name}  (downloads: {downloads:,})")
        # --- NEW: repo size check BEFORE any download ---
        size_gb = get_repo_size_gb(model_name)
        if size_gb is not None and size_gb > MAX_REPO_SIZE_GB:
            msg = f"Repo too large: {size_gb:.1f} GB > {MAX_REPO_SIZE_GB:.1f} GB"
            print(f"   - SKIP: {msg}")
            update_health(model_id, "TOO_LARGE", msg)
            tested += 1
            continue
        elif size_gb is not None:
            print(f"   Repo size (approx): {size_gb:.1f} GB")

        # dedicated per-model cache dir; easy to nuke after
        cache_dir = HF_HOME / f"cache_{model_id.replace('/','_').replace('@','_')}"
        cache_dir.mkdir(exist_ok=True)

        try:
            status, err = test_model_transformers(model_name, cache_dir)
            update_health(model_id, status, err)
            tag = "✓" if status == "OK" else ("⚠" if status in {"OOM","TRUST_NEEDED","TOO_LARGE"} else "✗")
            print(f"   {tag} {status}{(': ' + err) if err else ''}")
        except Exception as e:
            update_health(model_id, "ERROR", str(e))
            print(f"   ✗ ERROR: {str(e)[:200]}")
        finally:
            # per-model cache cleanup
            try:
                if cache_dir.exists(): shutil.rmtree(cache_dir, ignore_errors=True)
            except: pass
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        tested += 1

    # quick summary from DB
    print("\nSummary from DB:")
    cursor.execute("""
        SELECT health_status, COUNT(*)
        FROM Models
        WHERE problem=?
          AND library=?
          AND health_status IS NOT NULL
        GROUP BY health_status
        ORDER BY COUNT(*) DESC
    """, (PROBLEM, LIBRARY))
    for s, c in cursor.fetchall():
        print(f"  {s}: {c}")
    print(f"  Total processed in this run: {tested}")

if __name__ == "__main__":
    try:
        main()
    finally:
        try: conn.close()
        except: pass
        print("\nDone.")
