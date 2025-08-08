# RustLangExtract

Rust-native, schema-enforced structured extraction from **unstructured text** using **OpenAI (GPT-5)** with **JSON Schema** (Structured Outputs), then writes to **Parquet** and produces a simple **HTML viewer** highlighting extracted spans.

> Why this repo?
- **Strict structure**: Uses OpenAI “Structured Outputs” so the model must return schema-valid JSON.
- **Fast + safe**: All chunking/merging is native Rust; no Python runtime.
- **Analytics-ready**: Writes straight to Parquet (query with DuckDB/Polars).
- **Auditable**: Every extraction stores exact character spans and an HTML visualizer shows where it came from.

## Quick Start

### 1) Install Rust (stable)
https://rustup.rs

### 2) Set your OpenAI API key
```bash
export OPENAI_API_KEY="sk-...your key..."
