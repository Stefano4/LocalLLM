from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
from datetime import datetime
from pathlib import Path

import ollama
from flask import Flask, jsonify, send_from_directory, request as flask_request
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

#MODEL_NAME    = "qwen3-5-9b-local"        # Name assigned via 'ollama create'
MODEL_NAME    = "gemma4-e4b-8b-loc"        # Name assigned via 'ollama create'
OLLAMA_HOST   = "http://localhost:48085"  # Ollama daemon — managed by ollama_manager.sh
SERVER_PORT   = 48084                     # This Flask web server
INPUT_FOLDER  = "input"
OUTPUT_FOLDER = "output"
LOG_FOLDER    = "logs"
MAX_TOKENS    = 30000

# Single Ollama client instance, reused across all requests
_ollama_client = ollama.Client(host=OLLAMA_HOST)

SYSTEM_INSTRUCTION = (
    "Think through the problem carefully before answering.\n\n"
    "thinking — reason step by step: analyse the request, consider edge cases, "
    "plan your approach. Be thorough here.\n\n"
    "response — your final answer to the user. If you created files, "
    "mention each by name. Do NOT reproduce file content here.\n\n"
    "files — if the task requires any file, its full content MUST go here, "
    "not in 'response'. Rules:\n"
    "  • Use a descriptive name with the correct extension "
    "(.py, .sh, .md, .json, .csv, .html, etc.)\n"
    "  • Always write the complete, working content — never a stub, "
    "placeholder comment, or 'see above'.\n"
    "  • Set to [] if no file is needed."
)

# ─────────────────────────────────────────────────────────────────────────────
# Structured output schema
# ─────────────────────────────────────────────────────────────────────────────

class GeneratedFile(BaseModel):
    name: str
    content: str


class ModelOutput(BaseModel):
    thinking: str
    response: str
    files: list[GeneratedFile]   # no default → required in JSON schema → Ollama always emits it

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(timestamp: str, stem: str) -> logging.Logger:
    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_path = Path(LOG_FOLDER) / f"{timestamp}_{stem}.log"

    logger = logging.getLogger(f"local_llm.{timestamp}.{stem}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log → {log_path}")
    return logger

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def query_model(
    prompt_text: str,
    logger: logging.Logger,
) -> tuple[str, str, list[dict]]:
    """
    Send prompt_text to Ollama and return (raw_output, final_response, files).

    keep_alive=0 instructs Ollama to evict the model from VRAM as soon as
    the response is complete, freeing memory between requests.
    """
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user",   "content": prompt_text},
    ]

    sep = "─" * 72
    logger.debug(f"\n{sep}\nPROMPT\n{sep}\n{prompt_text}\n{sep}")

    try:
        raw_response = _ollama_client.chat(
            model=MODEL_NAME,
            messages=messages,
            format=ModelOutput.model_json_schema(),
            keep_alive=0,           # Evict model from VRAM after this request
            options={
                "num_predict": MAX_TOKENS,
                "temperature": 0.0,
            },
        )
        raw_output = raw_response.message.content
    except ollama.ResponseError as exc:
        logger.error(f"Ollama API error {exc.status_code}: {exc.error}")
        raise
    except Exception as exc:
        logger.error(f"Ollama call failed: {exc}")
        raise

    logger.debug(f"\n{sep}\nRAW OUTPUT\n{sep}\n{raw_output}\n{sep}")

    try:
        parsed = ModelOutput.model_validate_json(raw_output)
    except Exception as exc:
        logger.error(f"Structured output parse error: {exc}")
        return raw_output, raw_output, []

    if parsed.thinking:
        logger.debug(f"\n{sep}\nTHINKING\n{sep}\n{parsed.thinking}\n{sep}")

    safe_files: list[dict] = []
    for f in parsed.files:
        safe_name = Path(f.name).name
        if not safe_name or safe_name.startswith(".") or "/" in safe_name or "\\" in safe_name:
            logger.warning(f"Rejected unsafe file name from model: {f.name!r}")
            continue
        safe_files.append({"name": safe_name, "content": f.content})

    if safe_files:
        logger.info(f"Model produced {len(safe_files)} file(s): {[f['name'] for f in safe_files]}")

    return raw_output, parsed.response, safe_files

# ─────────────────────────────────────────────────────────────────────────────
# MIME helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    ext = Path(filename).suffix.lower()
    return _EXTRA_MIME.get(ext) or mimetypes.guess_type(filename)[0] or "application/octet-stream"

# ─────────────────────────────────────────────────────────────────────────────
# Output persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_markdown(prompt: str, response: str, timestamp: str, logger: logging.Logger) -> Path:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = Path(OUTPUT_FOLDER) / f"{timestamp}_response.md"
    human_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(
        f"# LLM Response\n\n"
        f"**Generated:** {human_ts}  \n"
        f"**Model:** {MODEL_NAME}\n\n"
        f"---\n\n"
        f"## Prompt\n\n{prompt}\n\n"
        f"---\n\n"
        f"## Response\n\n{response}\n",
        encoding="utf-8",
    )
    logger.info(f"Markdown → {path}")
    return path


def save_generated_files(files: list[dict], timestamp: str, logger: logging.Logger) -> list[dict]:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    saved = []
    for file_info in files:
        stored_name = f"{timestamp}_{file_info['name']}"
        dest = Path(OUTPUT_FOLDER) / stored_name
        dest.write_text(file_info["content"], encoding="utf-8")
        logger.info(f"Saved → {dest}")
        saved.append({
            "name":         file_info["name"],
            "stored_name":  stored_name,
            "path":         str(dest),
            "content":      file_info["content"],
            "mime_type":    _mime_type(file_info["name"]),
            "download_url": f"/files/{stored_name}",
        })
    return saved

# ─────────────────────────────────────────────────────────────────────────────
# Telegram payload builder
# ─────────────────────────────────────────────────────────────────────────────

_TG_CAPTION_LIMIT = 1024
_TG_MESSAGE_LIMIT = 4096

_MIME_EMOJI: list[tuple[str, str]] = [
    ("text/x-python",      "🐍"),
    ("text/javascript",    "📜"),
    ("text/typescript",    "📜"),
    ("text/x-shellscript", "🖥️"),
    ("text/markdown",      "📝"),
    ("text/html",          "🌐"),
    ("text/csv",           "📊"),
    ("application/json",   "🗂️"),
    ("application/xml",    "🗂️"),
    ("text/yaml",          "🗂️"),
    ("text/plain",         "📄"),
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
        reply = reply[: _TG_MESSAGE_LIMIT - 10] + "…"

    documents: list[dict] = []

    md_content = markdown_path.read_text(encoding="utf-8")
    documents.append(
        _make_document_entry(
            display_name="response.md",
            stored_name=markdown_path.name,
            content=md_content,
            mime_type="text/markdown",
            extra_caption="Full response with prompt",
        )
    )
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
        "reply":     reply,        # actual (possibly truncated) response text
        "documents": documents,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_data: dict, logger: logging.Logger, timestamp: str) -> dict:
    prompt = input_data.get("prompt", "").strip()
    if not prompt:
        raise ValueError("Input JSON must contain a non-empty 'prompt' key.")

    logger.info(f"Prompt: {prompt}")

    _, final_response, extracted_files = query_model(prompt, logger)

    markdown_path = save_markdown(prompt, final_response, timestamp, logger)
    saved_files   = save_generated_files(extracted_files, timestamp, logger)
    telegram      = build_telegram_payload(
        prompt, timestamp, final_response, markdown_path, saved_files
    )

    logger.info(f"=== Done {datetime.now().strftime('%H:%M:%S')} ===")

    return {
        "status":        "ok",
        "timestamp":     timestamp,
        "prompt":        prompt,
        "response":      final_response,
        "markdown_file": str(markdown_path),
        "files":         saved_files,
        "telegram":      telegram,
    }

# ─────────────────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────────────────

def run_server_mode() -> None:
    boot_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger  = setup_logger(boot_ts, "server_boot")
    logger.info(f"=== localLLM server — port {SERVER_PORT} | Ollama → {OLLAMA_HOST} ===")

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        ollama_reachable = False
        try:
            _ollama_client.list()
            ollama_reachable = True
        except Exception:
            pass
        return jsonify({
            "status":           "ok",
            "model":            MODEL_NAME,
            "ollama_host":      OLLAMA_HOST,
            "ollama_reachable": ollama_reachable,
            "max_tokens":       MAX_TOKENS,
        }), 200

    @app.route("/files/<path:filename>", methods=["GET"])
    def download_file(filename: str):
        safe_name  = Path(filename).name
        output_dir = Path(OUTPUT_FOLDER).resolve()
        if not (output_dir / safe_name).exists():
            return jsonify({"status": "error", "message": "File not found"}), 404
        return send_from_directory(
            str(output_dir), safe_name,
            as_attachment=True, mimetype=_mime_type(safe_name),
        )

    @app.route("/prompt", methods=["POST"])
    def handle_prompt():
        body = flask_request.get_json(force=True, silent=True)
        if not body:
            return jsonify({"status": "error", "message": "Invalid or missing JSON body"}), 400

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        req_logger = setup_logger(timestamp, "api_prompt")
        req_logger.info(f"POST /prompt — {flask_request.remote_addr}")

        try:
            result = run_pipeline(body, req_logger, timestamp)
            return jsonify(result), 200
        except ValueError as exc:
            req_logger.warning(f"Bad request: {exc}")
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:
            req_logger.error(f"Pipeline error: {exc}", exc_info=True)
            return jsonify({"status": "error", "message": str(exc)}), 500

    # Ollama handles its own request queue out-of-process, so threaded=True is safe
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, threaded=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLI file mode (local testing — reads first JSON in input/)
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
    logger.info(f"=== localLLM file mode — {timestamp} ===")
    logger.info(f"Input: {input_path}")

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
        description="localLLM — Ollama-powered LLM server with Telegram integration"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=f"Start HTTP server on port {SERVER_PORT} (default: process input/ folder)",
    )
    args = parser.parse_args()
    if args.serve:
        run_server_mode()
    else:
        run_file_mode()


if __name__ == "__main__":
    main()