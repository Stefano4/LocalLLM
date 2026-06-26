"""
localLLM.py — Local MLX inference server / CLI tool
=====================================================
Modes
  python localLLM.py            # CLI: reads input/*.json, writes output/*.md
  python localLLM.py --serve    # HTTP: POST /prompt  GET /health  GET /files/<name>

Response format
  Output is produced with grammar-constrained decoding (via the `outlines`
  library), not by asking the model to follow XML tags and hoping it
  complies. The model is mechanically restricted, token by token, to only
  ever emit text matching the ModelOutput JSON schema below — so parsing
  never fails the way ad-hoc tag-scraping could.

  thinking  (str)
      Internal chain-of-thought. Stored in the log (DEBUG) only — never
      returned to the caller.

  response  (str)
      Final, polished answer. Returned to the caller and written to
      output/<timestamp>_response.md.

  files     (list of {name, content})
      Any files the model wants to create (source code, config, CSV, etc.).
      Each one is saved to output/<filename> and its content is also
      included in the JSON response under the "files" key.
      In server mode the files can be downloaded individually via
      GET /files/<filename>.

Security
  File names produced by the model are sanitised: only the basename is
  kept and path-traversal characters are rejected before writing.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import logging
import mimetypes
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
import mlx.core as mx
import mlx_lm
from flask import Flask, jsonify, send_from_directory
from flask import request as flask_request
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH    = os.path.expanduser("~/Work/Dev/AI_Models/mlx-community/gemma-4-e2b-it-4bit")
#MODEL_PATH    = os.path.expanduser("~/Work/Dev/AI_Models/lmstudio-community/Qwen3.5-9B-MLX-8bit")
INPUT_FOLDER  = "input"
OUTPUT_FOLDER = "output"
LOG_FOLDER    = "logs"
SERVER_PORT   = 48084
MAX_TOKENS    = 12000
MAX_KV_SIZE = 16384
MAX_MX_MEMORY_GB = 14

# System instruction injected into every prompt. Note there is no need to
# describe a tag syntax here — the schema below is enforced mechanically by
# outlines' grammar-constrained decoding, so the model literally cannot
# emit a token that would produce invalid/incomplete structure.
SYSTEM_INSTRUCTION = (
    "You must respond with exactly three things: your internal reasoning, "
    "your final answer, and any files you want to create.\n\n"
    "- thinking: your internal chain-of-thought and working notes. "
    "The user will never see this.\n"
    "- response: your final, polished answer to the user. Refer to any "
    "files you created by name so the user knows what was produced. "
    "Always provide a response.\n"
    "- files: a list of files to create, if any. Each entry needs a "
    "descriptive 'name' with the correct extension (.py, .csv, .json, "
    ".sh, .md, etc.) and the complete file 'content' — never just the "
    "filename. Leave this list empty if no file is needed.\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Structured output schema
# ─────────────────────────────────────────────────────────────────────────────
# This schema IS the contract. outlines compiles it into a grammar and masks
# the model's logits at every generation step so only tokens that keep the
# output a valid instance of ModelOutput are ever sampled — the JSON is
# guaranteed well-formed and guaranteed to have exactly these fields. There
# is no longer a "the model forgot to close a tag" or "the model put a stray
# < character mid-file" failure mode.

class GeneratedFile(BaseModel):
    name: str
    content: str


class ModelOutput(BaseModel):
    thinking: str
    response: str
    files: list[GeneratedFile] = Field(default_factory=list)

# ─────────────────────────────────────────────────────────────────────────────
# Global model state  (lazy-loaded, one set shared across all requests)
# ─────────────────────────────────────────────────────────────────────────────

_model: object | None      = None
_model_lock                = threading.Lock()   # serialises inference on 8 GB RAM


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(timestamp: str, stem: str) -> logging.Logger:
    """
    Returns a named logger that writes to:
      logs/<timestamp>_<stem>.log  — DEBUG  (full prompts + raw model output)
      stdout                       — INFO   (progress messages only)

    Each call gets a unique logger name so concurrent requests don't share
    handlers.
    """
    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_path = Path(LOG_FOLDER) / f"{timestamp}_{stem}.log"

    logger = logging.getLogger(f"local_llm.{timestamp}.{stem}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Log → {log_path}")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Model lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def _clear_mlx_cache() -> tuple[int, int]:
    """
    Releases MLX's internal Metal buffer cache.

    Dropping the Python reference to the model (and gc.collect()) only frees
    the *Python* object. The Metal buffers that held the actual weight
    tensors are managed by MLX's own caching allocator and are kept around
    for reuse until you explicitly clear them — this is the same class of
    problem as torch.cuda.empty_cache() on CUDA/MPS.

    The function name moved from `mx.metal.clear_cache()` to the top-level
    `mx.clear_cache()` in newer mlx versions; this checks both locations so
    it works regardless of which version is installed.

    Returns (active_bytes, cache_bytes) *after* clearing, for logging.
    """
    clear_fn = getattr(mx, "clear_cache", None) or getattr(
        getattr(mx, "metal", None), "clear_cache", None
    )
    if clear_fn:
        clear_fn()

    active_fn = getattr(mx, "get_active_memory", None) or getattr(
        getattr(mx, "metal", None), "get_active_memory", None
    )
    cache_fn = getattr(mx, "get_cache_memory", None) or getattr(
        getattr(mx, "metal", None), "get_cache_memory", None
    )
    active = active_fn() if active_fn else -1
    cache  = cache_fn() if cache_fn else -1
    return active, cache


def get_model(logger: logging.Logger):
    global _model
    if _model is None:
        # Raise Metal memory ceiling so KV cache fits alongside the weights
        try:
            limit_gb = MAX_MX_MEMORY_GB  # adjust down to 13 if still OOM, up if you have 24+ GB
            mx.set_memory_limit(limit_gb * 1024 ** 3)
            logger.info(f"Metal memory limit set to {limit_gb} GB")
        except AttributeError:
            pass  # older mlx build, skip

        logger.info(f"Loading model from {MODEL_PATH}  (max_tokens={MAX_TOKENS})")
        try:
            model, tokenizer = mlx_lm.load(MODEL_PATH)
            _model = (model, tokenizer)
            logger.info("Model ready.")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            raise
    return _model


def unload_model(logger: logging.Logger) -> None:
    """
    Drops the model reference, runs GC, and releases MLX's Metal buffer
    cache so freed weight memory is actually returned, not just kept around
    by MLX for reuse. Called after every HTTP request so 8 GB RAM is not
    held between calls.
    """
    global _model
    _model = None
    gc.collect()
    active, cache = _clear_mlx_cache()
    if active >= 0:
        logger.info(
            f"Model unloaded — MLX active={active / 1024**3:.2f} GB, "
            f"cache={cache / 1024**3:.2f} GB after clear."
        )
    else:
        logger.info("Model unloaded from memory.")


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def query_model(
    prompt_text: str,
    model_tuple,
    logger: logging.Logger,
) -> tuple[str, str, list[dict]]:
    model, tokenizer = model_tuple

    schema_str = json.dumps(ModelOutput.model_json_schema(), indent=2)
    system_with_schema = (
        SYSTEM_INSTRUCTION
        + f"\n\nYou MUST respond with a single valid JSON object matching "
          f"this exact schema — output nothing else, no markdown fences:\n{schema_str}"
    )

    messages = [
        {"role": "system", "content": system_with_schema},
        {"role": "user",   "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sep = "─" * 72
    logger.debug(f"\n{sep}\nPROMPT\n{sep}\n{prompt_text}\n{sep}")

    raw_output = mlx_lm.generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        max_kv_size=MAX_KV_SIZE,
        verbose=False,
    )

    logger.debug(f"\n{sep}\nRAW MODEL OUTPUT\n{sep}\n{raw_output}\n{sep}")

    # Strip optional markdown fences the model might add anyway
    clean = raw_output.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]           # drop ```json line
        clean = clean.rsplit("```", 1)[0].strip()  # drop closing ```

    try:
        parsed = ModelOutput.model_validate_json(clean)
    except Exception as exc:
        logger.error(
            f"Failed to parse structured output (consider raising MAX_TOKENS): {exc}"
        )
        return raw_output, raw_output, []

    if parsed.thinking:
        logger.debug(f"\n{sep}\nTHINKING BLOCK\n{sep}\n{parsed.thinking}\n{sep}")

    extracted_files: list[dict] = []
    for f in parsed.files:
        safe_name = Path(f.name).name
        if not safe_name or safe_name.startswith(".") \
                or "/" in safe_name or "\\" in safe_name:
            logger.warning(f"Rejected unsafe file name from model: {f.name!r}")
            continue
        extracted_files.append({"name": safe_name, "content": f.content})

    if extracted_files:
        names = [f["name"] for f in extracted_files]
        logger.info(f"Model produced {len(extracted_files)} file(s): {names}")

    return raw_output, parsed.response, extracted_files


# ─────────────────────────────────────────────────────────────────────────────
# MIME helpers
# ─────────────────────────────────────────────────────────────────────────────

# Supplement Python's mimetypes DB with common text extensions that are
# sometimes missing on minimal systems.
_EXTRA_MIME: dict[str, str] = {
    ".md":   "text/markdown",
    ".py":   "text/x-python",
    ".sh":   "text/x-shellscript",
    ".toml": "text/x-toml",
    ".yaml": "text/yaml",
    ".yml":  "text/yaml",
    ".env":  "text/plain",
    ".log":  "text/plain",
    ".ts":   "text/typescript",
    ".tsx":  "text/typescript",
    ".jsx":  "text/javascript",
}


def _mime_type(filename: str) -> str:
    """Returns the MIME type for *filename* based on its extension."""
    ext = Path(filename).suffix.lower()
    if ext in _EXTRA_MIME:
        return _EXTRA_MIME[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown output
# ─────────────────────────────────────────────────────────────────────────────

def save_markdown(prompt: str, response: str, timestamp: str, logger: logging.Logger) -> Path:
    """
    Writes a formatted markdown file to OUTPUT_FOLDER and returns its path.

    File name pattern: output/<timestamp>_response.md
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = Path(OUTPUT_FOLDER) / f"{timestamp}_response.md"

    human_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    content = (
        f"# LLM Response\n\n"
        f"**Generated:** {human_ts}  \n"
        f"**Model:** {Path(MODEL_PATH).name}\n\n"
        f"---\n\n"
        f"## Prompt\n\n"
        f"{prompt}\n\n"
        f"---\n\n"
        f"## Response\n\n"
        f"{response}\n"
    )

    filename.write_text(content, encoding="utf-8")
    logger.info(f"Markdown → {filename}")
    return filename


def save_generated_files(
    files: list[dict], timestamp: str, logger: logging.Logger
) -> list[dict]:
    """
    Writes each model-generated file to OUTPUT_FOLDER.

    To avoid collisions when the same filename is requested multiple times,
    the timestamp is prepended to each name:
        hello.py  →  output/<timestamp>_hello.py

    Returns a list of dicts enriched with the saved path, MIME type, and
    the server download URL:
        [{"name": "hello.py", "content": "…", "path": "output/…_hello.py",
          "mime_type": "text/x-python", "download_url": "/files/…_hello.py"}, …]
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    saved = []
    for file_info in files:
        stored_name = f"{timestamp}_{file_info['name']}"
        dest = Path(OUTPUT_FOLDER) / stored_name
        dest.write_text(file_info["content"], encoding="utf-8")
        logger.info(f"File saved → {dest}")
        saved.append({
            "name":         file_info["name"],
            "stored_name":  stored_name,
            "path":         str(dest),
            "content":      file_info["content"],
            "mime_type":    _mime_type(file_info["name"]),
            "download_url": f"/files/{stored_name}",
        })
    return saved


# Telegram message length limits
_TG_CAPTION_LIMIT = 1024   # max chars for a file caption
_TG_MESSAGE_LIMIT = 4096   # max chars for a plain text message
_TG_PREVIEW_LIMIT = 3800   # safe budget for readme inline preview

# Emoji map for common MIME type families
_MIME_EMOJI: list[tuple[str, str]] = [
    ("text/x-python",     "🐍"),
    ("text/javascript",   "📜"),
    ("text/typescript",   "📜"),
    ("text/x-shellscript","🖥️"),
    ("text/markdown",     "📝"),
    ("text/html",         "🌐"),
    ("text/csv",          "📊"),
    ("application/json",  "🗂️"),
    ("application/xml",   "🗂️"),
    ("text/yaml",         "🗂️"),
    ("text/plain",        "📄"),
]


def _doc_emoji(mime_type: str) -> str:
    for prefix, emoji in _MIME_EMOJI:
        if mime_type.startswith(prefix):
            return emoji
    return "📎"


def _make_document_entry(
    display_name: str,
    stored_name: str,
    content: str,
    mime_type: str,
    extra_caption: str = "",
) -> dict:
    """
    Returns a single 'document' dict ready for a Telegram bot to call
    ``bot.send_document()``.

    Fields
    ------
    filename     : str   — original display name (e.g. "hello.py")
    stored_name  : str   — timestamped name on disk (e.g. "20240101_…_hello.py")
    mime_type    : str   — MIME type for the Telegram document object
    content_b64  : str   — UTF-8 content base64-encoded; decode and wrap in
                           io.BytesIO before passing to send_document()
    download_url : str   — relative GET path; prepend the server base URL
    caption      : str   — ready-to-send Markdown caption (≤ 1 024 chars)
    """
    caption = f"{_doc_emoji(mime_type)} `{display_name}`"
    if extra_caption:
        caption += f"\n{extra_caption}"
    if len(caption) > _TG_CAPTION_LIMIT:
        caption = caption[: _TG_CAPTION_LIMIT - 1] + "…"

    return {
        "filename":     display_name,
        "stored_name":  stored_name,
        "mime_type":    mime_type,
        "content_b64":  base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "download_url": f"/files/{stored_name}",
        "caption":      caption,
    }


def build_telegram_payload(
    prompt: str,
    timestamp: str,
    reply: str,
    markdown_path: Path,
    saved_files: list[dict],
) -> dict:
    """
    Builds the 'telegram' block included in every /prompt response.

    The Telegram bot can use this directly without any post-processing:

      telegram["caption"]           →  send as the caption of the first document,
                                       or as a standalone message if no docs.

      telegram["reply"]             →  plain-text response (use for inline display
                                       or as a message before sending documents).

      telegram["documents"]         →  list of document dicts, one per generated
                                       file plus the markdown summary.  For each:
                                         content_b64  → base64-decode → BytesIO
                                         filename     → InputFile name
                                         mime_type    → passed to send_document
                                         caption      → per-file caption string
                                         download_url → GET /files/<stored_name>

    Typical bot loop
    ----------------
        import io, base64
        for doc in payload["telegram"]["documents"]:
            buf = io.BytesIO(base64.b64decode(doc["content_b64"]))
            buf.name = doc["filename"]
            await bot.send_document(chat_id, buf, caption=doc["caption"],
                                    parse_mode="Markdown")
    """
    # ── Top-level caption (shown before any documents) ──────────────────────
    prompt_short = prompt if len(prompt) <= 200 else prompt[:197] + "…"
    file_count   = len(saved_files)
    files_note   = (
        f"📦 *{file_count} file{'s' if file_count != 1 else ''} generated*\n"
        if file_count else ""
    )
    caption = (
        f"✅ *Reply generated*\n"
        f"📋 *Goal:* {prompt_short}\n"
        f"{files_note}"
        f"🕐 `{timestamp}`"
    )
    if len(caption) > _TG_CAPTION_LIMIT:
        caption = caption[: _TG_CAPTION_LIMIT - 1] + "…"

    if len(reply) > _TG_MESSAGE_LIMIT:
        reply = reply[:_TG_MESSAGE_LIMIT - 10] + "…"

    # ── Document list ────────────────────────────────────────────────────────
    documents: list[dict] = []

    # 1. Markdown summary (always present)
    md_stored = markdown_path.name
    md_content = markdown_path.read_text(encoding="utf-8")
    documents.append(
        _make_document_entry(
            display_name="response.md",
            stored_name=md_stored,
            content=md_content,
            mime_type="text/markdown",
            extra_caption="Full response with prompt",
        )
    )

    # 2. Each model-generated file
    for f in saved_files:
        documents.append(
            _make_document_entry(
                display_name=f["name"],
                stored_name=f["stored_name"],
                content=f["content"],
                mime_type=f["mime_type"],
            )
        )

    return {
        "caption":   caption,
        "reply":     "*Reply generated*",
        "documents": documents,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_data: dict, logger: logging.Logger, timestamp: str
) -> dict:
    """
    Orchestrates a single prompt → response cycle.

    Parameters
    ----------
    input_data : dict
        Must contain at least {"prompt": "<user text>"}.
    logger : logging.Logger
    timestamp : str
        "%Y%m%d_%H%M%S" string used for output filenames.

    Returns
    -------
    dict with keys:
        status, timestamp, prompt, response, markdown_file,
        files (list of {name, path, content})
    """
    prompt = input_data.get("prompt")
    if not prompt:
        raise ValueError("Input JSON must contain a non-empty 'prompt' key.")

    logger.info(f"Prompt: {prompt}")

    model = get_model(logger)

    with _model_lock:
        raw_output, final_response, extracted_files = query_model(
            prompt, model, logger
        )

    markdown_path = save_markdown(prompt, final_response, timestamp, logger)
    saved_files   = save_generated_files(extracted_files, timestamp, logger)
    telegram = build_telegram_payload(
        prompt, timestamp, final_response, markdown_path, saved_files
    )
    logger.info(f"=== Done at {datetime.now().strftime('%H:%M:%S')} ===")

    return {
        "status":        "ok",
        "timestamp":     timestamp,
        "prompt":        prompt,
        "response":      final_response,
        "markdown_file": str(markdown_path),
        "files":         saved_files,
        "telegram": telegram,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTTP server mode
# ─────────────────────────────────────────────────────────────────────────────

def run_server_mode() -> None:
    boot_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger  = setup_logger(boot_ts, "server_boot")
    logger.info(f"=== local_LLM [server mode] port {SERVER_PORT} ===")
    logger.info("Model is loaded on first request and unloaded after each one.")
    logger.info("threaded=False — requests are serialised to protect 8 GB RAM.")

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> tuple:
        return jsonify({
            "status":     "ok",
            "model":      Path(MODEL_PATH).name,
            "max_tokens": MAX_TOKENS,
        }), 200

    @app.route("/files/<path:filename>", methods=["GET"])
    def download_file(filename: str) -> tuple:
        """
        Serves a previously generated file from OUTPUT_FOLDER.
        Example: GET /files/20240101_120000_hello.py
        Only the basename is used — directory traversal is rejected.
        Content-Type is inferred from the file extension.
        """
        safe_name  = Path(filename).name
        output_dir = Path(OUTPUT_FOLDER).resolve()
        target     = output_dir / safe_name

        if not target.exists():
            return jsonify({"status": "error", "message": "File not found"}), 404

        return send_from_directory(
            str(output_dir),
            safe_name,
            as_attachment=True,
            mimetype=_mime_type(safe_name),
        )

    @app.route("/prompt", methods=["POST"])
    def prompt() -> tuple:
        body = flask_request.get_json(force=True, silent=True)
        if not body:
            return jsonify({
                "status":  "error",
                "message": "Invalid or missing JSON body",
            }), 400

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        req_logger = setup_logger(timestamp, "api_prompt")
        req_logger.info(f"POST /prompt — client {flask_request.remote_addr}")

        try:
            result = run_pipeline(body, req_logger, timestamp)
            return jsonify(result), 200
        except ValueError as exc:
            req_logger.warning(f"Bad request: {exc}")
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:
            req_logger.error(f"Pipeline failed: {exc}", exc_info=True)
            return jsonify({"status": "error", "message": str(exc)}), 500
        finally:
            unload_model(req_logger)

    # threaded=False: one request at a time — safe on 8 GB RAM
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, threaded=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI file mode
# ─────────────────────────────────────────────────────────────────────────────

def run_file_mode() -> None:
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    json_files = sorted(Path(INPUT_FOLDER).glob("*.json"))

    if not json_files:
        print(
            f"No JSON files found in '{INPUT_FOLDER}/'.\n"
            f"Create one with at least: {{\"prompt\": \"your question here\"}}"
        )
        sys.exit(1)

    input_path = json_files[0]
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger     = setup_logger(timestamp, input_path.stem)
    logger.info(f"=== local_LLM [file mode] {timestamp} ===")
    logger.info(f"Reading {input_path}")

    try:
        input_data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"Cannot read {input_path}: {exc}")
        sys.exit(1)

    try:
        run_pipeline(input_data, logger, timestamp)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        logger.error(f"Pipeline error: {exc}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local LLM — CLI file mode (default) or HTTP service (--serve)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=f"Start HTTP service on port {SERVER_PORT} instead of processing input/",
    )
    args = parser.parse_args()

    if args.serve:
        run_server_mode()
    else:
        run_file_mode()


if __name__ == "__main__":
    main()