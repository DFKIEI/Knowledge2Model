import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import warnings
from PIL import Image, ImageDraw
from huggingface_hub import HfApi, hf_hub_download
import torch
import gc
import json

warnings.filterwarnings("ignore")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Config ---
HF_TOKEN = os.getenv("HF_TOKEN", "") # Path to your Hugging Face token
DATABASE_PATH = r"" # Path to your database
MIN_DOWNLOADS = 200
PROBLEM = "image-to-text"
SUPPORTED_LIB = "transformers"
MAX_REPO_SIZE_GB = float(os.getenv("MAX_REPO_SIZE_GB", "40"))

# --- DB ---
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


# ============================================================
# HELPER: Standard model loading kwargs
# ============================================================

def get_load_kwargs(device: int = 0, trust_remote_code: bool = False) -> dict:
    """Standard kwargs for model loading directly to GPU."""
    kwargs = {
        "torch_dtype": torch.float16,
        "device_map": {"": device},
        "low_cpu_mem_usage": True,
        "token": HF_TOKEN,
    }
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_model_architecture(model_name: str) -> str:
    """Fetch model architecture from config.json."""
    try:
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            token=HF_TOKEN,
        )
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Check various architecture indicators
        arch = config.get("architectures", [None])[0] if config.get("architectures") else None
        model_type = config.get("model_type", "")
        
        if arch:
            return arch.lower()
        return model_type.lower()
    except Exception:
        return ""


def repo_is_too_big(model_name: str, max_gb: float):
    """Check if repository size exceeds limit."""
    try:
        api = HfApi()
        info = api.repo_info(
            repo_id=model_name,
            repo_type="model",
            files_metadata=True,
            token=HF_TOKEN or None
        )
        bytes_total = 0
        for f in info.siblings or []:
            name = getattr(f, "rfilename", "") or getattr(f, "path", "") or ""
            size = getattr(f, "size", 0) or 0
            if isinstance(name, str) and any(name.endswith(ext) for ext in [".safetensors", ".bin", ".gguf", ".ckpt", ".pt"]):
                bytes_total += int(size)
        size_gb = float(bytes_total) / (1024 ** 3)
        return (size_gb >= max_gb, size_gb)
    except Exception:
        return (False, 0.0)


def check_repo_access(model_name: str) -> tuple:
    """Check if a repository is gated or inaccessible BEFORE downloading."""
    try:
        api = HfApi()
        try:
            api.list_repo_files(model_name, token=HF_TOKEN or None)
            return (True, None, "")
        except Exception as e:
            err_str = str(e).lower()
            if "gated" in err_str or "401" in err_str or "access" in err_str or "restricted" in err_str:
                return (False, "GATED", f"Gated repo - accept license at https://huggingface.co/{model_name}")
            if "404" in err_str or "not found" in err_str:
                return (False, "NOT_FOUND", "Repository not found")
            if "private" in err_str:
                return (False, "PRIVATE", "Private repository")
            raise e
    except Exception as e:
        err_str = str(e).lower()
        if "gated" in err_str or "401" in err_str or "access" in err_str:
            return (False, "GATED", f"Gated repo - accept license at https://huggingface.co/{model_name}")
        if "404" in err_str or "not found" in err_str:
            return (False, "NOT_FOUND", "Repository not found")
        return (True, None, "")


def make_test_image() -> Image.Image:
    """Create a simple test image."""
    img = Image.new('RGB', (384, 384), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 150], fill=(200, 100, 100))
    draw.rectangle([150, 200, 350, 300], fill=(100, 100, 200))
    draw.ellipse([250, 50, 350, 150], fill=(100, 200, 100))
    return img


def cleanup_gpu():
    """Thorough GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def has_meaningful_output(output, prompt: str = "") -> tuple:
    """Check if output contains meaningful generated text."""
    if output is None:
        return False, ""
    
    try:
        text = ""
        if isinstance(output, str):
            text = output
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            first = output[0]
            if isinstance(first, dict):
                text = str(first.get("generated_text") or first.get("caption") or first.get("text") or "")
            else:
                text = str(first)
        elif isinstance(output, dict):
            text = str(output.get("generated_text") or output.get("caption") or output.get("text") or "")
        else:
            text = str(output)
        
        for pattern in ["ASSISTANT:", "assistant:", "<|im_end|>", "</s>", "<s>", "<pad>"]:
            if pattern in text:
                text = text.split(pattern)[-1]
        
        text = text.strip()
        cleaned = ''.join(c for c in text if c.isalnum())
        return len(cleaned) >= 2, text
        
    except Exception:
        return False, ""


def classify_error(e: Exception) -> tuple:
    """Classify an exception into a status code and message."""
    err_str = str(e).lower()
    
    if "gated" in err_str or "access" in err_str or "401" in err_str or "restricted" in err_str:
        return "GATED", "Requires access approval on HuggingFace"
    if "out of memory" in err_str or "oom" in err_str or "cuda out of memory" in err_str:
        return "OOM", "Out of memory"
    if "no space left" in err_str or "not enough space" in err_str:
        return "DISK_FULL", "Disk full - insufficient space"
    if "404" in err_str or "not found" in err_str or "doesn't have any library metadata" in err_str:
        return "NOT_FOUND", "Model not found"
    if "you need to install" in err_str or "pip install" in err_str or "modulenotfounderror" in err_str:
        return "SKIP_DEP", f"Missing dependency: {str(e)[:100]}"
    if "fugashi" in err_str or "mecab" in err_str:
        return "SKIP_DEP", "Missing fugashi/MeCab for Japanese tokenization"
    if "does not appear to have a file named" in err_str:
        return "NO_WEIGHTS", "Model has no weights file"
    
    return None, str(e)


# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

def run_preflight_checks(model_name: str) -> tuple:
    """Run all pre-flight checks. Returns (can_proceed, status, error)."""
    
    model_name_lower = model_name.lower()
    
    # Check repo access
    is_accessible, error_status, error_msg = check_repo_access(model_name)
    if not is_accessible:
        return (False, error_status, error_msg)
    
    # Check repo size
    try:
        too_big, size_gb = repo_is_too_big(model_name, MAX_REPO_SIZE_GB)
        if too_big:
            return (False, "SKIP_LARGE", f"Repo {size_gb:.2f} GB >= {MAX_REPO_SIZE_GB:.2f} GB")
    except Exception:
        pass
    
    # Check CUDA
    if not torch.cuda.is_available():
        return (False, "NO_CUDA", "CUDA not available")
    
    # Skip AWQ/GPTQ models
    if any(q in model_name_lower for q in ["-awq", "_awq", "-gptq", "_gptq", "awq-int", "gptq-int"]):
        return (False, "SKIP_QUANT", "AWQ/GPTQ models require autoawq/auto-gptq")
    
    # Skip MLX models
    if "mlx-community" in model_name_lower or "/mlx" in model_name_lower:
        return (False, "SKIP_MLX", "MLX models are for Apple Silicon only")
    
    # Skip GGUF models
    if "-gguf" in model_name_lower or model_name_lower.endswith("-gguf"):
        return (False, "SKIP_GGUF", "GGUF models require llama.cpp")
    
    # Skip robotics/VLA models
    if "openvla" in model_name_lower:
        return (False, "SKIP_ROBOTICS", "Robotics/VLA model (outputs actions, not text)")
    
    # Skip embedding models
    if "vlm2vec" in model_name_lower:
        return (False, "SKIP_EMBEDDING", "Embedding model (outputs vectors, not text)")
    
    return (True, None, None)


# ============================================================
# SPECIALIZED HANDLERS
# ============================================================

def try_qwen2_vl(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle Qwen2-VL models (including OCR variants)."""
    print("    [S] Trying: Qwen2-VL handler...")
    
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device, trust_remote_code=True),
        )
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, token=HF_TOKEN,
        )
        
        # Qwen2-VL requires specific message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ]
        
        # Apply chat template
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        # Decode only the generated part
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_llava_onevision(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle LLaVA-OneVision models (VARCO-VISION, etc.)."""
    print("    [S] Trying: LLaVA-OneVision handler...")
    
    try:
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device, trust_remote_code=True),
        )
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, token=HF_TOKEN,
        )
        
        # LLaVA-OneVision uses chat format with image token
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        text = processor.decode(output_ids[0], skip_special_tokens=True)
        # Remove the prompt from output
        if "assistant" in text.lower():
            text = text.split("assistant")[-1].strip()
        
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_llava_next(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle LLaVA-NeXT models."""
    print("    [S] Trying: LLaVA-NeXT handler...")
    
    try:
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device, trust_remote_code=True),
        )
        processor = LlavaNextProcessor.from_pretrained(model_name, token=HF_TOKEN)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        text = processor.decode(output_ids[0], skip_special_tokens=True)
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_pix2struct(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle Pix2Struct models (DePlot, charts, documents)."""
    print("    [S] Trying: Pix2Struct handler...")
    
    try:
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
        
        model = Pix2StructForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device),
        )
        processor = Pix2StructProcessor.from_pretrained(model_name, token=HF_TOKEN)
        
        try:
            inputs = processor(images=image, text="Describe this image.", return_tensors="pt").to(model.device)
        except Exception:
            inputs = processor(images=image, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=128)
        
        text = processor.decode(output_ids[0], skip_special_tokens=True)
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_trocr(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle TrOCR models (true OCR, not VLM-based)."""
    print("    [S] Trying: TrOCR handler...")
    
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        processor = TrOCRProcessor.from_pretrained(model_name, token=HF_TOKEN)
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            **get_load_kwargs(device),
        )
        
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(model.device, dtype=torch.float16)
        
        with torch.inference_mode():
            output_ids = model.generate(pixel_values, max_new_tokens=64)
        
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        del model, processor, pixel_values, output_ids
        cleanup_gpu()
        
        # For OCR, empty on test image (no text) is OK
        return "OK", "OCR model functional (test image has no text)"
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_florence(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle Florence-2 models."""
    print("    [S] Trying: Florence-2 handler...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=HF_TOKEN,
            attn_implementation="eager",
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
        
        inputs = processor(text="<CAPTION>", images=image, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                num_beams=3,
            )
        
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_llava(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle standard LLaVA models."""
    print("    [S] Trying: LLaVA handler...")
    
    try:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device, trust_remote_code=True),
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
        
        # Try chat template first
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image briefly."},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception:
            prompt = "USER: <image>\nDescribe this image briefly.\nASSISTANT:"
        
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        text = processor.decode(output_ids[0], skip_special_tokens=True)
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_blip(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle BLIP models."""
    print("    [S] Trying: BLIP handler...")
    
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        
        model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device),
        )
        processor = BlipProcessor.from_pretrained(model_name, token=HF_TOKEN)
        
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64)
        
        text = processor.decode(output_ids[0], skip_special_tokens=True)
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_blip2(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle BLIP-2 models."""
    print("    [S] Trying: BLIP-2 handler...")
    
    try:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device),
        )
        processor = Blip2Processor.from_pretrained(model_name, token=HF_TOKEN)
        
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64)
        
        text = processor.decode(output_ids[0], skip_special_tokens=True)
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_kosmos(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle Kosmos-2 models."""
    print("    [S] Trying: Kosmos-2 handler...")
    
    try:
        from transformers import Kosmos2ForConditionalGeneration, AutoProcessor
        
        model = Kosmos2ForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device),
        )
        processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
        
        inputs = processor(text="<grounding>Describe this image:", images=image, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64)
        
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_idefics(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle Idefics2/3 models."""
    print("    [S] Trying: Idefics handler...")
    
    try:
        model_class = None
        try:
            from transformers import Idefics3ForConditionalGeneration
            model_class = Idefics3ForConditionalGeneration
        except ImportError:
            from transformers import Idefics2ForConditionalGeneration
            model_class = Idefics2ForConditionalGeneration
        
        from transformers import AutoProcessor
        
        model = model_class.from_pretrained(
            model_name,
            **get_load_kwargs(device, trust_remote_code=True),
        )
        processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
        
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64)
        
        text = processor.decode(output_ids[0], skip_special_tokens=True)
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_mllama(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle Llama-3.2-Vision models."""
    print("    [S] Trying: Mllama handler...")
    
    try:
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device),
        )
        processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
        
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image briefly."}]}]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=input_text, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        text = processor.decode(output[0], skip_special_tokens=True)
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_paligemma(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle PaliGemma models."""
    print("    [S] Trying: PaliGemma handler...")
    
    try:
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device),
        )
        processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
        
        inputs = processor(text="Describe this image.", images=image, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        text = processor.decode(output_ids[0], skip_special_tokens=True)
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_git(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle GIT (GenerativeImage2Text) models."""
    print("    [S] Trying: GIT handler...")
    
    try:
        from transformers import GitForCausalLM, GitProcessor
        
        model = GitForCausalLM.from_pretrained(
            model_name,
            **get_load_kwargs(device),
        )
        processor = GitProcessor.from_pretrained(model_name, token=HF_TOKEN)
        
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64)
        
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


def try_instructblip(model_name: str, image: Image.Image, device: int) -> tuple:
    """Handle InstructBLIP models."""
    print("    [S] Trying: InstructBLIP handler...")
    
    try:
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            **get_load_kwargs(device),
        )
        processor = InstructBlipProcessor.from_pretrained(model_name, token=HF_TOKEN)
        
        inputs = processor(images=image, text="Describe this image.", return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=64)
        
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        has_text, _ = has_meaningful_output(text)
        
        del model, processor, inputs, output_ids
        cleanup_gpu()
        
        return ("OK", "") if has_text else ("EMPTY", "No text generated")
        
    except Exception as e:
        cleanup_gpu()
        status, msg = classify_error(e)
        return (status, msg) if status else (None, str(e))


# ============================================================
# HANDLER ROUTING - ARCHITECTURE BASED
# ============================================================

def get_handlers_for_model(model_name: str, architecture: str) -> list:
    """Get handlers based on architecture and model name."""
    handlers = []
    name_lower = model_name.lower()
    arch_lower = architecture.lower()
    
    # Architecture-based routing (most reliable)
    if "qwen2vl" in arch_lower or "qwen2_vl" in arch_lower:
        handlers.append(("Qwen2-VL", try_qwen2_vl))
    
    if "llavaonevision" in arch_lower or "llava_onevision" in arch_lower:
        handlers.append(("LLaVA-OneVision", try_llava_onevision))
    
    if "llavanext" in arch_lower or "llava_next" in arch_lower:
        handlers.append(("LLaVA-NeXT", try_llava_next))
    
    if "llavaforconditional" in arch_lower:
        handlers.append(("LLaVA", try_llava))
    
    if "pix2struct" in arch_lower:
        handlers.append(("Pix2Struct", try_pix2struct))
    
    if "trocr" in arch_lower or "visionencoderdecoder" in arch_lower:
        # Only use TrOCR handler for actual TrOCR models
        if "trocr" in name_lower or ("ocr" in name_lower and "qwen" not in name_lower and "llava" not in name_lower):
            handlers.append(("TrOCR", try_trocr))
    
    if "florence" in arch_lower:
        handlers.append(("Florence", try_florence))
    
    if "blip2" in arch_lower or "blip-2" in arch_lower:
        handlers.append(("BLIP-2", try_blip2))
    
    if "blipforconditional" in arch_lower and "blip2" not in arch_lower:
        handlers.append(("BLIP", try_blip))
    
    if "kosmos" in arch_lower:
        handlers.append(("Kosmos", try_kosmos))
    
    if "idefics" in arch_lower:
        handlers.append(("Idefics", try_idefics))
    
    if "mllama" in arch_lower:
        handlers.append(("Mllama", try_mllama))
    
    if "paligemma" in arch_lower:
        handlers.append(("PaliGemma", try_paligemma))
    
    if "git" in arch_lower and "gitforconditional" in arch_lower:
        handlers.append(("GIT", try_git))
    
    if "instructblip" in arch_lower:
        handlers.append(("InstructBLIP", try_instructblip))
    
    # Name-based fallbacks (if architecture didn't match)
    if not handlers:
        if "qwen" in name_lower and ("vl" in name_lower or "vision" in name_lower):
            handlers.append(("Qwen2-VL", try_qwen2_vl))
        
        if "varco" in name_lower or "onevision" in name_lower:
            handlers.append(("LLaVA-OneVision", try_llava_onevision))
        
        if "llava-next" in name_lower or "llava_next" in name_lower:
            handlers.append(("LLaVA-NeXT", try_llava_next))
        
        if "llava" in name_lower:
            handlers.append(("LLaVA", try_llava))
        
        if "florence" in name_lower:
            handlers.append(("Florence", try_florence))
        
        if "blip2" in name_lower or "blip-2" in name_lower:
            handlers.append(("BLIP-2", try_blip2))
        
        if "blip" in name_lower and "blip2" not in name_lower:
            handlers.append(("BLIP", try_blip))
        
        if "pix2struct" in name_lower or "deplot" in name_lower or "matcha" in name_lower:
            handlers.append(("Pix2Struct", try_pix2struct))
        
        if "kosmos" in name_lower:
            handlers.append(("Kosmos", try_kosmos))
        
        if "idefics" in name_lower:
            handlers.append(("Idefics", try_idefics))
        
        if "paligemma" in name_lower:
            handlers.append(("PaliGemma", try_paligemma))
        
        if "git-" in name_lower or "/git-" in name_lower:
            handlers.append(("GIT", try_git))
        
        if "instructblip" in name_lower:
            handlers.append(("InstructBLIP", try_instructblip))
    
    return handlers


# ============================================================
# MAIN TESTING FUNCTION
# ============================================================

def try_image_to_text(model_name: str) -> tuple:
    """Test if a VLM works. Returns (status, error_message)."""
    
    print("    Checking prerequisites...")
    can_proceed, status, error = run_preflight_checks(model_name)
    if not can_proceed:
        return status, error
    
    # Get model architecture
    print("    Detecting architecture...")
    architecture = get_model_architecture(model_name)
    print(f"    Architecture: {architecture or 'unknown'}")
    
    device = 0
    image = make_test_image()
    
    # Get handlers based on architecture
    handlers = get_handlers_for_model(model_name, architecture)
    
    if handlers:
        print(f"    Found {len(handlers)} specialized handler(s): {[h[0] for h in handlers]}")
        
        all_errors = []
        for handler_name, handler_func in handlers:
            try:
                status, error = handler_func(model_name, image, device)
                if status is not None:
                    return status, error
                all_errors.append(f"{handler_name}: {error[:80] if error else 'unknown'}")
            except Exception as e:
                err_str = str(e)
                all_errors.append(f"{handler_name}: {err_str[:80]}")
                
                if "out of memory" in err_str.lower():
                    cleanup_gpu()
                    return "OOM", "Out of memory"
                if "no space left" in err_str.lower():
                    cleanup_gpu()
                    return "DISK_FULL", "Disk full"
                cleanup_gpu()
        
        # If all specialized handlers failed, try generic pipeline
        print("    Specialized handlers failed, trying generic pipeline...")
    
    # Try generic pipeline as fallback
    from transformers import pipeline
    
    print("    [1] Trying: image-to-text pipeline...")
    try:
        pipe = pipeline(
            "image-to-text",
            model=model_name,
            device=device,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        result = pipe(image, max_new_tokens=50)
        has_text, text = has_meaningful_output(result)
        del pipe
        cleanup_gpu()
        if has_text:
            print(f"        ✓ Success: {text[:50]}...")
            return "OK", ""
        else:
            print(f"        Empty output")
    except Exception as e:
        cleanup_gpu()
        err_str = str(e)
        print(f"        Failed: {err_str[:80]}...")
        
        status, msg = classify_error(e)
        if status in ["GATED", "OOM", "DISK_FULL", "NOT_FOUND", "NO_WEIGHTS", "SKIP_DEP"]:
            return status, msg
    
    print("    [2] Trying: image-text-to-text pipeline...")
    try:
        pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            device=device,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        result = pipe(image, text="Describe this image.", max_new_tokens=50)
        has_text, text = has_meaningful_output(result)
        del pipe
        cleanup_gpu()
        if has_text:
            print(f"        ✓ Success: {text[:50]}...")
            return "OK", ""
        else:
            print(f"        Empty output")
            return "EMPTY", "All methods returned empty output"
    except Exception as e:
        cleanup_gpu()
        err_str = str(e)
        print(f"        Failed: {err_str[:80]}...")
        
        status, msg = classify_error(e)
        if status:
            return status, msg
    
    return "FAIL", "All handlers and pipelines failed"


# ============================================================
# DATABASE & MAIN LOOP
# ============================================================

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
    
    print("=" * 60)
    print("IMAGE-TO-TEXT MODEL HEALTH CHECK")
    print("=" * 60)
    print(f"Models to test: {remaining}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except:
        pass
    
    print("=" * 60)

    i = 0
    while True:
        row = fetch_next()
        if not row:
            print("\nNo more models to test.")
            break

        model_id, model_name, downloads, library = row
        i += 1
        print(f"\n[{i}/{remaining}] {model_name}")
        print(f"    Downloads: {downloads:,}")

        try:
            status, err = try_image_to_text(model_name)
            update_health(model_id, status, err)
            
            if status == "OK":
                print(f"    ✓ OK")
            else:
                print(f"    ✗ {status}: {err[:80] if err else ''}")
                
        except Exception as e:
            error_str = str(e)
            status, msg = classify_error(e)
            if not status:
                status = "FAIL"
                msg = error_str[:500]
            update_health(model_id, status, msg)
            print(f"    ✗ {status}: {msg[:80]}")
            
        finally:
            cleanup_gpu()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    cur.execute("""
        SELECT health_status, COUNT(*)
          FROM Models
         WHERE problem=? AND health_status IS NOT NULL
         GROUP BY health_status
         ORDER BY COUNT(*) DESC
    """, (PROBLEM,))
    for status, cnt in cur.fetchall():
        print(f"  {status}: {cnt}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        conn.close()
        print("\nDatabase connection closed.")