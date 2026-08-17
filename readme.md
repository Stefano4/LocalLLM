# LocalLLM

A lightweight, self-hosted bridge between [Ollama](https://ollama.com) and your own automations. `localLLM.py` sends prompts to a local model through Ollama, forces the model to return **structured JSON** (reasoning, response text, and any generated files), saves everything to disk, and optionally exposes it all as a small Flask HTTP API — with a payload shape ready to drop straight into a Telegram bot.

It can run two ways:

- **File mode** — drop a JSON file with a `prompt` into `input/`, run the script, get a response and any generated files in `output/`.
- **Server mode** (`--serve`) — run as a long-lived HTTP service that accepts prompts over `POST /prompt`, useful for wiring up to bots, webhooks, or other apps.

---

## How it works

1. A prompt is submitted (via file or HTTP request).
2. The script sends it to a local Ollama model, along with a system instruction that forces the model to answer using a strict JSON schema with three fields:
   - `thinking` — the model's internal reasoning (logged, not shown to the user).
   - `response` — the final answer text.
   - `files` — a list of `{name, content}` objects for any files the task required (code, scripts, markdown, etc.).
3. The structured output is validated with [Pydantic](https://docs.pydantic.dev/).
4. The response is saved as a Markdown file, and any generated files are written to `output/` with a timestamp prefix.
5. A Telegram-ready payload (caption, truncated reply text, and base64-encoded document attachments) is built automatically.
6. Everything is logged to `logs/` (per-request log files, both to console and to disk).

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running, with a model already created/pulled
- Python packages:
  - `ollama`
  - `flask`
  - `pydantic`

Install dependencies:

```bash
pip install ollama flask pydantic
```

> There is currently no `requirements.txt` in the repo — the above three packages are all that's needed.

---

## Setup

### 1. Prepare your Ollama model

The script expects a model to already exist under a specific name in Ollama (see `MODEL_NAME` in the configuration section of `localLLM.py`). Pull or create your model with a matching name, for example:

```bash
ollama create gemma4-e4b-8b-loc -f Modelfile
```

or point `MODEL_NAME` in the script at any model you already have (`ollama list` to see what's available).

### 2. Configure the script

Open `localLLM.py` and adjust the constants at the top if needed:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `"gemma4-e4b-8b-loc"` | The Ollama model tag to query |
| `OLLAMA_HOST` | `http://localhost:48085` | Address of the Ollama daemon |
| `SERVER_PORT` | `48084` | Port for the Flask server (`--serve` mode) |
| `INPUT_FOLDER` | `input` | Where file-mode looks for `*.json` prompts |
| `OUTPUT_FOLDER` | `output` | Where responses and generated files are saved |
| `LOG_FOLDER` | `logs` | Where per-run log files are written |
| `MAX_TOKENS` | `30000` | `num_predict` passed to Ollama |

Note the non-default `OLLAMA_HOST` port (`48085`) — if you're running Ollama with its normal defaults (`11434`), update this constant or set your Ollama daemon to listen on `48085`.

### 3. Make sure the folders exist

`input/`, `output/`, and `logs/` are created automatically on first run if they don't already exist.

---

## Usage

### File mode (default)

Create a JSON file inside `input/` with at least a `prompt` key:

```json
{
  "prompt": "Write a Python script that renames all .txt files in a folder to lowercase."
}
```

Then run:

```bash
python localLLM.py
```

The script picks up the **first** `*.json` file (alphabetically) found in `input/`, sends the prompt to the model, and writes:

- `output/<timestamp>_response.md` — the full prompt + response as Markdown
- `output/<timestamp>_<filename>` — any files the model generated
- `logs/<timestamp>_<input-stem>.log` — a detailed log of the run (prompt, raw model output, reasoning, errors)

If no JSON files are found in `input/`, the script prints a usage hint and exits.

### Server mode

Start the HTTP server:

```bash
python localLLM.py --serve
```

This starts a Flask app on `0.0.0.0:<SERVER_PORT>` (default `48084`).

#### Endpoints

**`GET /health`**

Returns server status and whether the Ollama daemon is reachable.

```json
{
  "status": "ok",
  "model": "gemma4-e4b-8b-loc",
  "ollama_host": "http://localhost:48085",
  "ollama_reachable": true,
  "max_tokens": 30000
}
```

**`POST /prompt`**

Send a JSON body with a `prompt` field:

```bash
curl -X POST http://localhost:48084/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize the plot of Dune in 3 sentences."}'
```

Response:

```json
{
  "status": "ok",
  "timestamp": "20260720_143012",
  "prompt": "Summarize the plot of Dune in 3 sentences.",
  "response": "...",
  "markdown_file": "output/20260720_143012_response.md",
  "files": [
    {
      "name": "example.py",
      "stored_name": "20260720_143012_example.py",
      "path": "output/20260720_143012_example.py",
      "content": "...",
      "mime_type": "text/x-python",
      "download_url": "/files/20260720_143012_example.py"
    }
  ],
  "telegram": {
    "caption": "✅ *Reply generated*\n📋 *Goal:* Summarize the plot of Dune in 3 sentences.\n🕐 `20260720_143012`",
    "reply": "...",
    "documents": [ { "filename": "response.md", "content_b64": "...", "caption": "..." } ]
  }
}
```

The `telegram` object is pre-formatted for a Telegram bot: a caption with emoji, the reply text (truncated to Telegram's 4096-character limit), and a `documents` array with base64-encoded file content, MIME-type-based emoji, and captions truncated to Telegram's 1024-character caption limit — ready to be sent via `sendDocument`/`sendMessage` calls.

**`GET /files/<filename>`**

Downloads a previously generated file from `output/` by its stored (timestamped) filename.

---

## Output files

Every run produces a Markdown record of the exchange:

```markdown
# LLM Response

**Generated:** 2026-07-20 14:30:12
**Model:** gemma4-e4b-8b-loc

---

## Prompt

<the prompt text>

---

## Response

<the model's final response>
```

Any files the model decides to generate (scripts, configs, notes, etc.) are saved alongside it with the same timestamp prefix, and their MIME type is inferred from the file extension (with extra mappings for `.md`, `.py`, `.sh`, `.toml`, `.yaml`/`.yml`, `.ts`/`.tsx`, `.jsx`, etc.).

Generated filenames are sanitized before saving — path separators, leading dots, or empty names are rejected to prevent writing outside `output/`.

---

## Logging

Every run (boot, file-mode run, or individual API request) gets its own log file under `logs/`, named `<timestamp>_<context>.log`, containing:

- The full prompt sent to the model
- The raw JSON returned by Ollama
- The model's reasoning (`thinking` field), if present
- Any parsing or API errors

Logs are written at `DEBUG` level to file and `INFO` level to the console.

---

## Project structure

```
LocalLLM/
├── localLLM.py       # Main script — file mode and server mode
├── input/             # Drop JSON prompt files here for file mode
├── output/            # Generated responses (.md) and files (created at runtime)
└── logs/               # Per-run logs (created at runtime)
```

---

## Notes & caveats

- `keep_alive=0` is passed on every Ollama call, so the model is evicted from VRAM immediately after each response — good for memory-constrained machines running multiple models, at the cost of a reload delay on the next request.
- `temperature` is fixed at `0.0` for deterministic output.
- File mode always processes only the **first** JSON file found in `input/` per run — it does not batch-process the whole folder.
- The Flask server runs with `threaded=True` and `debug=False`; Ollama manages its own request queue out-of-process.
- There is no authentication on the `/prompt` or `/files` endpoints — if you expose the server beyond `localhost`, put it behind a reverse proxy or add your own auth layer.

---

## License

No license file is currently included in this repository. Add one (e.g. MIT) if you intend for others to reuse this code.