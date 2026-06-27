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
from flask import Flask, jsonify, send_from_directory
from flask import request as flask_request
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Il nome assegnato durante il comando 'ollama create'
MODEL_NAME    = "qwen3-5-9b-local"

INPUT_FOLDER  = "input"
OUTPUT_FOLDER = "output"
LOG_FOLDER    = "logs"
SERVER_PORT   = 48084
MAX_TOKENS    = 12000

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

class GeneratedFile(BaseModel):
    name: str
    content: str


class ModelOutput(BaseModel):
    thinking: str
    response: str
    files: list[GeneratedFile] = Field(default_factory=list)

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
# Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

def query_model(
    prompt_text: str,
    logger: logging.Logger,
) -> tuple[str, str, list[dict]]:

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user",   "content": prompt_text},
    ]

    sep = "─" * 72
    logger.debug(f"\n{sep}\nPROMPT\n{sep}\n{prompt_text}\n{sep}")

    try:
        # Sfrutta il vincolo di formato nativo di Ollama passando lo schema JSON di Pydantic
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            format=ModelOutput.model_json_schema(),
            options={
                "num_predict": MAX_TOKENS,
                "temperature": 0.0  # Zero per garantire la massima stabilità strutturale
            }
        )
        raw_output = response.message.content
    except Exception as exc:
        logger.error(f"Ollama API call failed: {exc}")
        raise

    logger.debug(f"\n{sep}\nRAW MODEL OUTPUT\n{sep}\n{raw_output}\n{sep}")

    try:
        parsed = ModelOutput.model_validate_json(raw_output)
    except Exception as exc:
        logger.error(f"Failed to parse structured output validation: {exc}")
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
    if ext in _EXTRA_MIME:
        return _EXTRA_MIME[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"

# ─────────────────────────────────────────────────────────────────────────────
# Markdown output
# ─────────────────────────────────────────────────────────────────────────────

def save_markdown(prompt: str, response: str, timestamp: str, logger: logging.Logger) -> Path:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = Path(OUTPUT_FOLDER) / f"{timestamp}_response.md"

    human_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    content = (
        f"# LLM Response\n\n"
        f"**Generated:** {human_ts}  \n"
        f"**Model:** {MODEL_NAME}\n\n"
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


_TG_CAPTION_LIMIT = 1024
_TG_MESSAGE_LIMIT = 4096

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
        reply = reply[:_TG_MESSAGE_LIMIT - 10] + "…"

    documents: list[dict] = []

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
    prompt = input_data.get("prompt")
    if not prompt:
        raise ValueError("Input JSON must contain a non-empty 'prompt' key.")

    logger.info(f"Prompt: {prompt}")

    # Ollama gestisce le richieste in modo sicuro e isolato
    raw_output, final_response, extracted_files = query_model(prompt, logger)

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
        "telegram":      telegram,
    }

# ─────────────────────────────────────────────────────────────────────────────
# HTTP server mode
# ─────────────────────────────────────────────────────────────────────────────

def run_server_mode() -> None:
    boot_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger  = setup_logger(boot_ts, "server_boot")
    logger.info(f"=== local_LLM [Ollama Server Mode] port {SERVER_PORT} ===")

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> tuple:
        return jsonify({
            "status":     "ok",
            "model":      MODEL_NAME,
            "max_tokens": MAX_TOKENS,
        }), 200

    @app.route("/files/<path:filename>", methods=["GET"])
    def download_file(filename: str) -> tuple:
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

    # Ora threaded può essere True senza rischi: Ollama gestisce la coda delle richieste out-of-process
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, threaded=True)

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
    logger.info(f"=== local_LLM [Ollama File Mode] {timestamp} ===")
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
        description="Local LLM via Ollama Engine — CLI file mode or HTTP service (--serve)"
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