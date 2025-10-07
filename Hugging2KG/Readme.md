## Hugging2KG Pipeline (Updated)

### Purpose
Crawl HuggingFace API, extract model metadata using LLM, and prepare data for Knowledge Graph construction.

---

### Pipeline Flow

#### **Stage 1: Data Collection**
1. **`hugging2sql.py`** - API Crawler
   - Crawls HuggingFace API for ML models
   - Stores raw model data in SQLite database
   - **Output**: `huggingface2.db`
#### **Stage 2: Model Card Extraction**
2. **`extract_model_cards.py`** - README Fetcher
   - Fetches README.md (model cards) from HuggingFace for each model
   - Stores complete model cards in database
   - **Updates**: `huggingface2.db` (adds `model_card` column)

#### **Stage 3: LLM-Based Tag Extraction**
3. **`model_card_tag_extractor.py`** - Intelligent Tag Extraction
   - Uses LLM (via LM Studio) to extract structured technical tags from model cards
   - Generates both `model_card_tags` and `metrics` separately
   - **Updates**: `huggingface2.db` (adds `model_card_tags` and `metrics` columns)

#### **Stage 4: Tag Cleaning & Validation**
4. **`filterInvalidTags.py`** - Tag Quality Control
   - Cleans and validates LLM-extracted tags
   - **Updates**: `huggingface2.db` (cleans `model_card_tags` and `metrics`)

---

### Supporting Files (Currently Used)

- **`metric_mapping.json`** - Normalizes metric names across different formats
  - Example: "acc", "accuracy", "acc/f1" → "Accuracy"
  - Used by downstream scripts for metric standardization

### Supporting Files (Not Currently Used)

- **`modality_mapping.json`** - Maps problem types to input/output modalities (legacy)

---

## Workflow Summary

**Complete pipeline (run in order):**
```bash
1. python hugging2sql.py                  # Crawl HuggingFace API → creates DB
2. python extract_model_cards.py          # Fetch model cards → adds model_card column
3. python model_card_tag_extractor.py     # LLM extracts tags → adds model_card_tags & metrics
4. python filterInvalidTags.py            # Clean/validate tags → final cleanup