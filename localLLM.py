"""
localLLM.py — Local MLX inference server / CLI tool
=====================================================
Modes
  python localLLM.py            # CLI: reads input/*.json, writes output/*.md
  python localLLM.py --serve    # HTTP: POST /prompt  GET /health  GET /files/<name>

Response format
  The model is instructed to use three XML tag types:

  <thinking> … </thinking>
      Internal chain-of-thought.  Stored in the log (DEBUG) only — never
      returned to the caller.

  <response> … </response>
      Final, polished answer.  Returned to the caller and written to
      output/response_<timestamp>.md.

  <file name="filename.ext"> … </file>   (zero or more)
      Any file the model wants to create (source code, config, CSV, etc.).
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
import gc
import json
import logging
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

import mlx_lm
from flask import Flask, jsonify, send_from_directory
from flask import request as flask_request

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH    = "/Volumes/TOSHIBA_NEW/HomeServer/AI Models/models/mlx-community/gemma-4-e2b-it-4bit"
INPUT_FOLDER  = "input"
OUTPUT_FOLDER = "output"
LOG_FOLDER    = "logs"
SERVER_PORT   = 48084
MAX_TOKENS    = 12000

# Delimiters the model is instructed to use
TAG_THINKING  = ("thinking", "<thinking>",  "</thinking>")
TAG_RESPONSE  = ("response", "<response>",  "</response>")

# System instruction injected into every prompt so the model structures output
SYSTEM_INSTRUCTION = (
    "Always structure your reply using these XML tags — and only these tags:\n\n"
    "<thinking>\n"
    "Your internal chain-of-thought, reasoning, and working notes go here.\n"
    "The user will never see this section.\n"
    "</thinking>\n\n"
    "<response>\n"
    "Your final, polished answer to the user goes here.\n"
    "Refer to any files you created by name so the user knows what was produced.\n"
    "</response>\n\n"
    "If the user asks you to create one or more files, also include a block like "
    "this for EACH file — placed after </response>:\n\n"
    '<file name="example.py">\n'
    "example.py\n"
    "</file>\n\n"
    "If no file is needed, omit the <file> block entirely.\n\n"
    "Do not include any text outside the XML blocks described above."
)

# ─────────────────────────────────────────────────────────────────────────────
# Global model state  (lazy-loaded, one set shared across all requests)
# ─────────────────────────────────────────────────────────────────────────────

_model: object | None      = None
_tokenizer: object | None  = None
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

def get_model(logger: logging.Logger) -> tuple:
    """Lazy-loads the model on first call; returns (model, tokenizer)."""
    global _model, _tokenizer
    if _model is None:
        logger.info(f"Loading model from {MODEL_PATH}  (max_tokens={MAX_TOKENS})")
        try:
            _model, _tokenizer = mlx_lm.load(MODEL_PATH)
            logger.info("Model ready.")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            raise
    return _model, _tokenizer


def unload_model(logger: logging.Logger) -> None:
    """
    Drops model + tokenizer references and runs GC.
    Called after every HTTP request so 8 GB RAM is not held between calls.
    """
    global _model, _tokenizer
    _model = _tokenizer = None
    gc.collect()
    logger.info("Model unloaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _build_messages(prompt_text: str) -> list[dict]:
    """Wraps the user prompt with the system instruction."""
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user",   "content": prompt_text},
    ]


def _extract_tag(raw: str, tag_name: str) -> str | None:
    """
    Returns the content of the first <tag_name>…</tag_name> block found in
    *raw*, or None if the tag is absent.  Whitespace around the content is
    stripped.
    """
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, raw, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_files(raw: str) -> list[dict]:
    """
    Finds every  <file name="…">…</file>  block in *raw* and returns a list
    of {"name": <safe filename>, "content": <text>} dicts.

    The filename is sanitised: only the basename is kept and any entry that
    still contains path-separator characters is skipped with a warning.
    """
    pattern = r'<file\s+name="([^"]+)">(.*?)</file>'
    results = []
    for raw_name, content in re.findall(pattern, raw, re.DOTALL):
        safe_name = Path(raw_name).name          # drop any directory prefix
        if "/" in safe_name or "\\" in safe_name or safe_name.startswith("."):
            continue                             # reject remaining traversal attempts
        results.append({"name": safe_name, "content": content.strip()})
    return results


def query_model(
    prompt_text: str,
    model,
    tokenizer,
    logger: logging.Logger,
) -> tuple[str, str, list[dict]]:
    """
    Runs one inference pass.

    Returns
    -------
    raw_output : str
        The complete, unmodified model output (logged at DEBUG).
    final_response : str
        Text extracted from the <response> block, or the full raw output if
        the model did not follow the structured format.
    extracted_files : list[dict]
        Zero or more {"name": str, "content": str} dicts from <file> blocks.
    """
    messages = _build_messages(prompt_text)

    # Some tokenizers do not accept a system role; gracefully degrade
    try:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        combined = f"{SYSTEM_INSTRUCTION}\n\n{prompt_text}"
        full_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": combined}],
            tokenize=False,
            add_generation_prompt=True,
        )

    sep = "─" * 72
    logger.debug(f"\n{sep}\nPROMPT\n{sep}\n{full_prompt}\n{sep}")

    raw_output = mlx_lm.generate(
        model, tokenizer, prompt=full_prompt, max_tokens=MAX_TOKENS
    )

    logger.debug(f"\n{sep}\nRAW MODEL OUTPUT\n{sep}\n{raw_output}\n{sep}")

    thinking = _extract_tag(raw_output, TAG_THINKING[0])
    if thinking:
        logger.debug(f"\n{sep}\nTHINKING BLOCK\n{sep}\n{thinking}\n{sep}")

    final_response = _extract_tag(raw_output, TAG_RESPONSE[0])
    if final_response is None:
        logger.warning(
            "Model output did not contain a <response> block. "
            "Returning full output as the response."
        )
        final_response = raw_output

    extracted_files = _extract_files(raw_output)
    if extracted_files:
        names = [f["name"] for f in extracted_files]
        logger.info(f"Model produced {len(extracted_files)} file(s): {names}")

    return raw_output, final_response, extracted_files


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

    Returns a list of dicts enriched with the saved path:
        [{"name": "hello.py", "content": "…", "path": "output/…_hello.py"}, …]
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    saved = []
    for file_info in files:
        dest = Path(OUTPUT_FOLDER) / f"{timestamp}_{file_info['name']}"
        dest.write_text(file_info["content"], encoding="utf-8")
        logger.info(f"File saved → {dest}")
        saved.append({
            "name":    file_info["name"],
            "path":    str(dest),
            "content": file_info["content"],
        })
    return saved


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

    model, tokenizer = get_model(logger)

    with _model_lock:
        raw_output, final_response, extracted_files = query_model(
            prompt, model, tokenizer, logger
        )

    markdown_path = save_markdown(prompt, final_response, timestamp, logger)
    saved_files   = save_generated_files(extracted_files, timestamp, logger)

    logger.info(f"=== Done at {datetime.now().strftime('%H:%M:%S')} ===")

    return {
        "status":        "ok",
        "timestamp":     timestamp,
        "prompt":        prompt,
        "response":      final_response,
        "markdown_file": str(markdown_path),
        "files":         saved_files,
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
        """
        safe_name = Path(filename).name
        output_dir = Path(OUTPUT_FOLDER).resolve()
        target     = output_dir / safe_name

        if not target.exists():
            return jsonify({"status": "error", "message": "File not found"}), 404

        return send_from_directory(
            str(output_dir), safe_name, as_attachment=True
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