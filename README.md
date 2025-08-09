# RustLangExtract

Rust-native, schema-enforced structured extraction from unstructured text.
Calls OpenAI’s **Responses API** with strict JSON Schema (Structured Outputs), writes results to **Parquet** for analytics, and renders a lightweight HTML viewer with **exact-span highlights**.

This version supports **multi-pass extraction, voting/union logic, adaptive re-chunking, cross-chunk consolidation, validation, and optional LLM-driven repair**, making it highly effective for **long, complex documents** in domains such as healthcare, finance, and law.

Inspired by and conceptually aligned with the excellent Python project [LangExtract by Google](https://github.com/google/langextract#installation).

---

## Why RustLangExtract

* Strict schema enforcement via OpenAI Structured Outputs (`text.format: { type: "json_schema", ... }`).
* Multi-pass extraction with union or majority voting.
* Parallel/concurrent chunk processing.
* Adaptive re-chunking of boundary “hot spots.”
* Cross-chunk consolidation with IoU-based deduplication.
* Regex/domain-specific validation for formats like email, ZIP, ICD-10, date, and currency.
* Optional LLM-driven repair for invalid extractions.
* Analytics-ready: Arrow2 + Parquet output for instant DuckDB/Polars queries.
* Auditable: stores exact character spans; HTML viewer shows precise highlights.

---

## Installation

### Prerequisites

* Rust (stable): install via [https://rustup.rs](https://rustup.rs)
* An OpenAI API key in your environment: `OPENAI_API_KEY`

### Build

```bash
git clone <your_repo_url>
cd RustLangExtract
cargo build --release
```

### Install to Cargo bin (optional)

```bash
cargo install --path .
# Ensure ~/.cargo/bin is on your PATH
```

---

## Usage

### Quick demo (embedded multi-domain sample)

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

---

## Common Flags

* `--model <id>` — OpenAI model ID (default: `gpt-5`)
* `--chunk-size <n>` / `--overlap <n>` — chunking controls
* `--passes <n>` — number of passes per chunk for self-consistency
* `--vote <union|majority>` — voting strategy for multi-pass results
* `--majority-threshold <n>` — threshold for `majority` voting
* `--second-pass` — enable cross-chunk dedupe/merge
* `--adaptive-rechunk` — reprocess boundary hot spots
* `--hotspot-threshold <n>` — % near edges to trigger re-chunk
* `--rechunk-size <n>` / `--rechunk-overlap <n>` — size/overlap for re-chunking
* `--validator` — enable validation checks
* `--validator-retry` — attempt LLM repair on invalid extractions
* `--preset <name>` — built-in presets (`medical`, `finance`, `legal`)
* `--prompt-file <path>` — append custom instructions from file
* `--classes <csv>` — extraction classes to prioritize
* `--concurrency <n>` — max parallel chunk requests
* `--max-retries <n>` — retry count for API calls
* `--timeout-seconds <n>` — per-request timeout (default 120)
* `--out-dir <dir>` — output directory (default `./out`)
* `--demo` — run with embedded multi-domain sample

---

## Outputs

* **Parquet** — `out/extractions.parquet` (Arrow2 schema)
* **JSONL** — `out/extractions.jsonl`
* **HTML** — `out/view.html` (click rows to jump to highlighted spans)

---

## What Gets Extracted

This tool guides the model to extract concrete entities across domains (e.g., people, organizations, dates, amounts, medications, ICD-10 codes, legal clauses, addresses, invoice lines, stock trades, sports stats, course metadata).

Enforced schema (simplified):

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
          "attributes": {"type": "object"}
        }
      }
    }
  },
  "required": ["document_id", "chunk_id", "extractions"]
}
```

---

## How It Works

1. Reads text (from `--input` file or `--demo`)
2. Chunks with overlap for boundary safety
3. Runs N passes per chunk, merging with union/majority voting
4. Optional adaptive re-chunking on boundary hot spots
5. Optional global consolidation of extractions
6. Optional validation and LLM repair
7. Writes Parquet + JSONL + HTML viewer

---

## Tips

* Increase `--passes` for more robust extraction in noisy text.
* Use `--second-pass` for multi-chunk documents to remove duplicates.
* Combine `--validator` with `--validator-retry` for compliance-sensitive contexts.
* Inspect results in `out/view.html` or query Parquet in DuckDB:

```sql
SELECT extraction_class, COUNT(*)
FROM 'out/extractions.parquet'
GROUP BY 1
ORDER BY 2 DESC;
```

---

## Credits & Attribution

* Inspired by Google’s LangExtract (schema-driven extraction, examples-guided prompting)
* Independent Rust re-implementation using Arrow2/Parquet and OpenAI’s Responses API
* Not affiliated with Google

---

## License

Apache-2.0 (same spirit as the referenced project). See `LICENSE`.
