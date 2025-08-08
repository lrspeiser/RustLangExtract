# RustLangExtract

Rust-native, schema-enforced structured extraction from unstructured text. Calls OpenAI’s Responses API with strict JSON Schema (Structured Outputs), writes results to Parquet, and renders a lightweight HTML viewer with highlighted spans.

Inspired by and conceptually aligned with the excellent Python project LangExtract by Google. For Python users, notebooks, and richer visualization, see LangExtract’s docs and install guide: [github.com/google/langextract#installation](https://github.com/google/langextract#installation).

## Why RustLangExtract?
- Strict JSON structure via OpenAI Structured Outputs (Responses API with `text.format: { type: "json_schema", ... }`).
- Fast and safe: all chunking/IO in Rust; no Python runtime required.
- Analytics‑ready: Arrow2 + Parquet output for instant DuckDB/Polars queries.
- Auditable: stores exact character spans; HTML viewer shows precise highlights.

## Installation

### Prerequisites
- Rust (stable): install via https://rustup.rs
- An OpenAI API key in your environment: `OPENAI_API_KEY`

### Build
```bash
git clone <your_repo_url>
cd RustLangExtract
cargo build --release
```

Optionally, install to your cargo bin:
```bash
cargo install --path .
# Ensure ~/.cargo/bin is on your PATH
```

## Usage

### Quick demo (embedded multi‑domain sample)
```bash
# via cargo
cargo run --release -- --demo --out-dir ./out --timeout-seconds 120

# or run the built binary
target/release/rustlangextract --demo --out-dir ./out --timeout-seconds 120
```

### Run on your own file
```bash
cargo run --release -- --input ./example.txt --out-dir ./out --timeout-seconds 180
```

### Common flags
- `--model <id>`: OpenAI model ID (default: `gpt-5`).
- `--chunk-size <n>` / `--overlap <n>`: chunking controls.
- `--concurrency <n>`: max parallel chunk requests.
- `--max-retries <n>`: retry count for API calls.
- `--timeout-seconds <n>`: per-request timeout (default 120).
- `--out-dir <dir>`: output directory (default `./out`).
- `--demo`: run with embedded multi‑domain sample (no input file needed).

### Outputs
- Parquet: `out/extractions.parquet` (Arrow2 schema)
- JSONL:   `out/extractions.jsonl`
- HTML:    `out/view.html` (click rows to jump to highlighted spans)

## What gets extracted?

This tool guides the model to extract concrete entities across domains (e.g., people, organizations, dates, amounts, medications, ICD‑10 codes, legal clauses, addresses, invoice lines, stock trades, sports stats, course metadata). It enforces the following schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "document_id": {"type": "string"},
    "chunk_id": {"type": "string"},
    "extractions": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "extraction_class": {"type": "string"},
          "extraction_text": {"type": "string"},
          "start_char": {"type": "integer"},
          "end_char": {"type": "integer"},
          "attributes": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "unit": {"type": "string"},
              "currency": {"type": "string"},
              "role": {"type": "string"},
              "dose": {"type": "string"},
              "frequency": {"type": "string"},
              "route": {"type": "string"},
              "code": {"type": "string"},
              "description": {"type": "string"},
              "jurisdiction": {"type": "string"},
              "qty": {"type": "number"},
              "unit_price": {"type": "number"},
              "line_total": {"type": "number"},
              "amount": {"type": "number"},
              "low": {"type": "number"},
              "high": {"type": "number"},
              "timestamp": {"type": "string"},
              "side": {"type": "string"},
              "symbol": {"type": "string"},
              "address": {"type": "string"},
              "city": {"type": "string"},
              "state": {"type": "string"},
              "zip": {"type": "string"},
              "phone": {"type": "string"},
              "email": {"type": "string"},
              "product": {"type": "string"},
              "organization": {"type": "string"},
              "person": {"type": "string"},
              "type": {"type": "string"}
            }
          }
        },
        "required": ["extraction_class", "extraction_text", "start_char", "end_char", "attributes"]
      }
    }
  },
  "required": ["document_id", "chunk_id", "extractions"]
}
```

## How it works (high level)
1. Reads text (from `--input` file or `--demo`).
2. Chunks with overlap for boundary safety.
3. Calls OpenAI Responses API with:
   - strict JSON Schema (Structured Outputs)
   - a few‑shot example to improve recall/precision
4. Shifts local chunk offsets to global.
5. Writes Parquet + JSONL + HTML viewer.

## Tips
- If requests time out or rate-limit, increase `--timeout-seconds` (e.g., 180) and reduce `--concurrency`.
- You can switch models via `--model <id>`.
- For quick inspection, open `out/view.html`. For analytics, query the Parquet file in DuckDB:
  ```sql
  SELECT extraction_class, COUNT(*) FROM 'out/extractions.parquet' GROUP BY 1 ORDER BY 2 DESC;
  ```

## Credits & Attribution
- This project was inspired by Google’s LangExtract and borrows ideas around schema‑driven extraction and examples‑guided prompting. See: [github.com/google/langextract#installation](https://github.com/google/langextract#installation).
- RustLangExtract is an independent re‑implementation in Rust, using Arrow2/Parquet and the OpenAI Responses API for schema‑enforced outputs. It is not affiliated with Google.

## License
Apache‑2.0 (same spirit as the referenced project). See `LICENSE`.


