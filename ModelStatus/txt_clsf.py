import os
import sqlite3
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ===== Config =====
HF_TOKEN = os.getenv("HF_TOKEN", "") # Path to your Hugging Face token
DATABASE_PATH = r"" # Path to your database
PROBLEM = "token-classification"
MAX_REPO_SIZE_GB = 40
MIN_DOWNLOADS = 15

# Libraries to test (in order) - SpaCy removed as it requires pip installation
SUPPORTED_LIBS = ["transformers", "gliner", "flair", "stanza"]

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
    """Return total repo size in GB. If lookup fails, return 0.0 (don't block)."""
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

# ===== Test Texts =====
TEST_TEXT_EN = (
    "Barack Obama was born in Hawaii and served in the White House. "
    "Apple hired Tim Cook as CEO in 2011, based in Cupertino."
)

TEST_TEXT_DE = (
    "Angela Merkel wurde in Hamburg geboren und arbeitete in Berlin. "
    "Siemens stellte Joe Kaeser als CEO ein."
)

TEST_TEXT_FR = (
    "Emmanuel Macron est né à Amiens et travaille à Paris. "
    "L'Oréal a embauché Jean-Paul Agon comme PDG."
)

TEST_TEXT_ES = (
    "Pablo Picasso nació en Málaga y vivió en Barcelona. "
    "Telefónica contrató a José María Álvarez como CEO."
)

TEST_TEXT_ZH = (
    "李明在北京大学学习，后来在上海的阿里巴巴公司工作。"
)

TEST_TEXT_AR = (
    "ولد محمد في القاهرة وعمل في شركة أرامكو في الرياض."
)

TEST_TEXT_PT = (
    "Cristiano Ronaldo nasceu em Funchal e jogou no Real Madrid. "
    "A Petrobras contratou Pedro Silva como diretor."
)

TEST_TEXT_IT = (
    "Leonardo da Vinci nacque a Vinci e lavorò a Milano. "
    "Ferrari ha assunto Marco Rossi come CEO."
)

TEST_TEXT_NL = (
    "Vincent van Gogh werd geboren in Zundert en werkte in Amsterdam. "
    "Philips heeft Jan de Vries aangenomen als CEO."
)

TEST_TEXT_RU = (
    "Владимир Путин родился в Ленинграде и работал в Москве. "
    "Газпром нанял Ивана Петрова директором."
)

TEST_TEXT_JA = (
    "田中太郎は東京で生まれ、ソニーで働いています。"
)

TEST_TEXT_KO = (
    "김철수는 서울에서 태어나 삼성에서 일하고 있습니다."
)

def get_test_text_for_model(model_name: str) -> str:
    """Return appropriate test text based on model name."""
    model_lower = model_name.lower()
    
    # German
    if any(x in model_lower for x in ["-de", "german", "deutsch", "-de-", "_de_", "de_core"]):
        return TEST_TEXT_DE
    # French
    elif any(x in model_lower for x in ["-fr", "french", "francais", "-fr-", "_fr_", "fr_core", "camembert"]):
        return TEST_TEXT_FR
    # Spanish
    elif any(x in model_lower for x in ["-es", "spanish", "espanol", "-es-", "_es_", "es_core", "beto"]):
        return TEST_TEXT_ES
    # Chinese
    elif any(x in model_lower for x in ["-zh", "chinese", "-zh-", "_zh_", "zh_core", "bert-base-chinese", "ckip"]):
        return TEST_TEXT_ZH
    # Arabic
    elif any(x in model_lower for x in ["-ar", "arabic", "-ar-", "_ar_", "camel", "arabert"]):
        return TEST_TEXT_AR
    # Portuguese
    elif any(x in model_lower for x in ["-pt", "portuguese", "-pt-", "_pt_", "pt_core", "bertimbau"]):
        return TEST_TEXT_PT
    # Italian
    elif any(x in model_lower for x in ["-it", "italian", "-it-", "_it_", "it_core"]):
        return TEST_TEXT_IT
    # Dutch
    elif any(x in model_lower for x in ["-nl", "dutch", "-nl-", "_nl_", "nl_core"]):
        return TEST_TEXT_NL
    # Russian
    elif any(x in model_lower for x in ["-ru", "russian", "-ru-", "_ru_", "ru_core", "rubert"]):
        return TEST_TEXT_RU
    # Japanese
    elif any(x in model_lower for x in ["-ja", "japanese", "-ja-", "_ja_", "ja_core", "tohoku"]):
        return TEST_TEXT_JA
    # Korean
    elif any(x in model_lower for x in ["-ko", "korean", "-ko-", "_ko_", "ko_core", "klue"]):
        return TEST_TEXT_KO
    
    return TEST_TEXT_EN


# =============================================================================
# TRANSFORMERS TEST
# =============================================================================
def test_transformers(model_name: str):
    """Test a transformers library model."""
    # Size gate
    size_gb = repo_size_gb(model_name)
    if size_gb and size_gb > MAX_REPO_SIZE_GB:
        return "SKIP_LARGE", f"Repo {size_gb:.2f} GB > {MAX_REPO_SIZE_GB} GB"

    import torch
    from transformers import pipeline

    test_text = get_test_text_for_model(model_name)
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        nlp = pipeline(
            "token-classification",
            model=model_name,
            device=device,
            token=HF_TOKEN,
            aggregation_strategy="simple",
            trust_remote_code=True
        )
        out = nlp(test_text)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == 0:
            torch.cuda.empty_cache()
            nlp = pipeline(
                "token-classification",
                model=model_name,
                device=-1,
                token=HF_TOKEN,
                aggregation_strategy="simple",
                trust_remote_code=True
            )
            out = nlp(test_text)
        else:
            raise

    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict) and any(k in first for k in ("entity_group", "entity", "label")):
            return "OK", ""
    return "FAIL", "No entities produced."


# =============================================================================
# GLINER TEST
# =============================================================================
def test_gliner(model_name: str):
    """Test a GLiNER library model."""
    # Size gate
    size_gb = repo_size_gb(model_name)
    if size_gb and size_gb > MAX_REPO_SIZE_GB:
        return "SKIP_LARGE", f"Repo {size_gb:.2f} GB > {MAX_REPO_SIZE_GB} GB"

    from gliner import GLiNER

    test_text = get_test_text_for_model(model_name)
    
    # GLiNER requires entity labels to extract
    labels = ["person", "organization", "location", "date", "company", "city", "country", "name", "place"]
    
    model = GLiNER.from_pretrained(model_name)
    entities = model.predict_entities(test_text, labels)
    
    if entities and len(entities) > 0:
        if isinstance(entities[0], dict) and "label" in entities[0]:
            return "OK", ""
    return "FAIL", "No entities produced."


# =============================================================================
# FLAIR TEST
# =============================================================================
def test_flair(model_name: str):
    """Test a Flair library model."""
    # Size gate
    size_gb = repo_size_gb(model_name)
    if size_gb and size_gb > MAX_REPO_SIZE_GB:
        return "SKIP_LARGE", f"Repo {size_gb:.2f} GB > {MAX_REPO_SIZE_GB} GB"

    # Force CPU to avoid CUDA errors
    import flair
    flair.device = "cpu"
    
    from flair.data import Sentence
    from flair.nn import Classifier
    
    test_text = get_test_text_for_model(model_name)
    
    # Load model from HuggingFace hub
    tagger = Classifier.load(model_name)
    
    sentence = Sentence(test_text)
    tagger.predict(sentence)
    
    # Check for NER entities
    ner_spans = sentence.get_spans("ner")
    if ner_spans and len(ner_spans) > 0:
        return "OK", ""
    
    # Check for POS tags
    pos_spans = sentence.get_spans("pos")
    if pos_spans and len(pos_spans) > 0:
        return "OK", ""
    
    # Check for any labels on tokens
    for token in sentence:
        if token.get_labels():
            return "OK", ""
    
    # Check for chunk tags (noun phrases)
    np_spans = sentence.get_spans("np")
    if np_spans and len(np_spans) > 0:
        return "OK", ""
    
    # Check for frame tags
    frame_spans = sentence.get_spans("frame")
    if frame_spans and len(frame_spans) > 0:
        return "OK", ""
    
    # Try getting all spans with any label type
    all_spans = sentence.get_spans()
    if all_spans and len(all_spans) > 0:
        return "OK", ""
    
    return "FAIL", "No entities produced."


# =============================================================================
# STANZA TEST
# =============================================================================
def test_stanza(model_name: str):
    """Test a Stanza library model with CPU to avoid CUDA errors."""
    # Size gate
    size_gb = repo_size_gb(model_name)
    if size_gb and size_gb > MAX_REPO_SIZE_GB:
        return "SKIP_LARGE", f"Repo {size_gb:.2f} GB > {MAX_REPO_SIZE_GB} GB"

    import stanza
    
    test_text = get_test_text_for_model(model_name)
    
    # Extract language code from model name
    # Format: "stanfordnlp/stanza-en" -> "en", "stanfordnlp/stanza-zh-hans" -> "zh-hans"
    if "/" in model_name:
        model_part = model_name.split("/")[-1]  # "stanza-en"
    else:
        model_part = model_name
    
    # Extract language code
    if model_part.startswith("stanza-"):
        lang_code = model_part.replace("stanza-", "")
    else:
        lang_code = model_part
    
    # Download the model
    try:
        stanza.download(lang_code, verbose=False)
    except Exception as e:
        return "FAIL", f"Failed to download stanza model for '{lang_code}': {str(e)[:100]}"
    
    # Load with CPU to avoid CUDA errors
    try:
        nlp = stanza.Pipeline(lang_code, verbose=False, use_gpu=False)
    except Exception as e:
        return "FAIL", f"Failed to load stanza pipeline for '{lang_code}': {str(e)[:100]}"
    
    doc = nlp(test_text)
    
    # Check for NER entities
    for sentence in doc.sentences:
        # Check sentence-level entities
        if hasattr(sentence, 'ents') and sentence.ents:
            return "OK", ""
        
        # Check token-level NER tags
        for token in sentence.tokens:
            if hasattr(token, 'ner') and token.ner and token.ner != 'O':
                return "OK", ""
        
        # Check word-level NER (some models use this)
        for word in sentence.words:
            if hasattr(word, 'ner') and word.ner and word.ner != 'O':
                return "OK", ""
    
    # Check for POS tags (most stanza models do POS tagging)
    for sentence in doc.sentences:
        for word in sentence.words:
            if hasattr(word, 'pos') and word.pos:
                return "OK", ""
            if hasattr(word, 'upos') and word.upos:
                return "OK", ""
            if hasattr(word, 'xpos') and word.xpos:
                return "OK", ""
    
    # Check for lemmas
    for sentence in doc.sentences:
        for word in sentence.words:
            if hasattr(word, 'lemma') and word.lemma and word.lemma != word.text:
                return "OK", ""
    
    # Check for dependency parsing
    for sentence in doc.sentences:
        for word in sentence.words:
            if hasattr(word, 'deprel') and word.deprel:
                return "OK", ""
    
    return "FAIL", "No entities or POS tags produced."


# =============================================================================
# DISPATCHER
# =============================================================================
def test_model(model_name: str, library: str):
    """Dispatch to the appropriate test function based on library."""
    if library == "transformers":
        return test_transformers(model_name)
    elif library == "gliner":
        return test_gliner(model_name)
    elif library == "flair":
        return test_flair(model_name)
    elif library == "stanza":
        return test_stanza(model_name)
    else:
        return "SKIP", f"Unsupported library: {library}"


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================
def fetch_next(library: str):
    cur.execute("""
        SELECT model_id, model_name, downloads, library
          FROM Models
         WHERE problem=? AND library=?
           AND health_status IS NULL
           AND downloads >= ?
         ORDER BY downloads DESC
         LIMIT 1
    """, (PROBLEM, library, MIN_DOWNLOADS))
    return cur.fetchone()

def count_remaining(library: str):
    cur.execute("""
        SELECT COUNT(*) FROM Models
         WHERE problem=? AND library=?
           AND health_status IS NULL
           AND downloads >= ?
    """, (PROBLEM, library, MIN_DOWNLOADS))
    return cur.fetchone()[0]

def count_by_library():
    """Show count of untested models per library."""
    print("\nUntested models by library:")
    for lib in SUPPORTED_LIBS:
        count = count_remaining(lib)
        print(f"  {lib}: {count}")
    print()

def test_library(library: str):
    """Test all models for a specific library."""
    remaining = count_remaining(library)
    if remaining == 0:
        print(f"No untested {library} models found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing {library.upper()} models ({remaining} remaining)")
    print(f"{'='*60}")
    
    i = 0
    while True:
        row = fetch_next(library)
        if not row:
            print(f"No more {library} models to test.")
            break

        model_id, model_name, downloads, lib = row
        i += 1
        print(f"[{i}] Testing: {model_name} | downloads: {downloads:,} | lib: {lib}")

        try:
            status, err = test_model(model_name, library)
            update_health(model_id, status, err)
            print("   ->", "OK" if status == "OK" else f"{status}: {err[:160]}")
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "does not exist" in msg:
                update_health(model_id, "NOT_FOUND", str(e))
                print("   -> NOT_FOUND:", str(e)[:100])
            elif "out of memory" in msg or "oom" in msg:
                update_health(model_id, "OOM", "Out of memory")
                print("   -> OOM: Out of memory")
            elif "gated" in msg or "access" in msg and "403" in msg:
                update_health(model_id, "GATED", str(e))
                print("   -> GATED:", str(e)[:100])
            else:
                update_health(model_id, "FAIL", str(e)[:500])
                print("   -> FAIL:", str(e)[:200])
        finally:
            # Clear GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

def print_summary():
    """Print summary of all tested models."""
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    cur.execute("""
        SELECT health_status, COUNT(*)
          FROM Models
         WHERE problem=?
           AND health_status IS NOT NULL
        GROUP BY health_status
        ORDER BY COUNT(*) DESC
    """, (PROBLEM,))
    
    total = 0
    for status, cnt in cur.fetchall():
        print(f"  {status}: {cnt}")
        total += cnt
    print(f"  TOTAL TESTED: {total}")

def reset_library_status(library: str):
    """Reset health status for a specific library (for re-testing)."""
    cur.execute("""
        UPDATE Models
           SET health_status = NULL, health_error = NULL, last_checked = NULL
         WHERE problem = ? AND library = ?
    """, (PROBLEM, library))
    conn.commit()
    print(f"Reset {cur.rowcount} models for library: {library}")

def main():
    ensure_columns()
    
    print("="*60)
    print("TOKEN CLASSIFICATION HEALTH CHECK")
    print(f"Libraries: {', '.join(SUPPORTED_LIBS)}")
    print(f"Config: MIN_DOWNLOADS={MIN_DOWNLOADS}, MAX_REPO_SIZE_GB={MAX_REPO_SIZE_GB}")
    print("="*60)
    
    count_by_library()
    
    # Test each library
    for library in SUPPORTED_LIBS:
        try:
            test_library(library)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Stopping...")
            break
        except Exception as e:
            print(f"Error testing {library}: {e}")
    
    print_summary()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        conn.close()
        print("\nDatabase connection closed.")