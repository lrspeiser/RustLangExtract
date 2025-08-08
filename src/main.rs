#![recursion_limit = "256"]
//! RustLangExtract
//! ----------------
//! Standalone Rust tool to extract structured data from large text files using
//! OpenAI (GPT-5) with **Structured Outputs (JSON Schema)**.
//!
//! Pipeline:
//!   input.txt -> chunk -> call LLM -> strict JSON -> merge spans -> Parquet + JSONL + HTML viewer
//!
//! Major design goals:
//!  - SAFE: no Python, memory-safe chunking, strict schema outputs
//!  - AUDITABLE: every extraction includes exact source spans (start_char/end_char)
//!  - ANALYTICS-READY: writes Parquet for instant DuckDB/Polars querying
//!
//! Console logging: very verbose so you can follow step-by-step.

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use reqwest::Client;
use ropey::Rope;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cmp::min;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
// use time::OffsetDateTime; // unused
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

// ================================
// CLI + Config
// ================================

#[derive(Debug, Parser)]
#[command(name="rustlangextract", version, about="Rust-native structured extraction with Parquet + HTML viewer")]
struct Cli {
    /// Input UTF-8 text file
    #[arg(long, value_name="FILE", required_unless_present = "demo")]
    input: Option<PathBuf>,

    /// Run with a built-in multi-domain demo corpus
    #[arg(long, default_value_t = false, conflicts_with = "input")]
    demo: bool,

    /// Output directory (Parquet, JSONL, HTML)
    #[arg(long, value_name="DIR", default_value="./out")]
    out_dir: PathBuf,

    /// OpenAI model ID (e.g., gpt-5.0-mini, gpt-5.0, etc.)
    #[arg(long, default_value="gpt-5")]
    model: String,

    /// Characters per chunk
    #[arg(long, default_value_t=10_000)]
    chunk_size: usize,

    /// Overlap between chunks (to avoid boundary misses)
    #[arg(long, default_value_t=500)]
    overlap: usize,

    /// Max retries for API calls
    #[arg(long, default_value_t=3)]
    max_retries: usize,

    /// Max parallel chunk requests (default: num_cpus)
    #[arg(long)]
    concurrency: Option<usize>,

    /// Overall HTTP request timeout in seconds (default: 120)
    #[arg(long)]
    timeout_seconds: Option<u64>,

    /// Optional file containing a custom prompt/instructions to control what to extract
    #[arg(long, value_name="FILE")]
    prompt_file: Option<PathBuf>,

    /// Comma-separated list of classes to extract (e.g., "character,emotion,relationship")
    #[arg(long, value_name="CSV")]
    classes: Option<String>,

    /// Built-in preset of instructions (e.g., medical, finance, legal)
    #[arg(long, value_name="NAME")]
    preset: Option<String>,
}

#[derive(Debug, Clone)]
struct Config {
    document_id: String,
    model: String,
    chunk_size: usize,
    overlap: usize,
    max_retries: usize,
    concurrency: usize,
    out_dir: PathBuf,
    timeout_seconds: u64,
}

// ================================
// Schema: What we want the model to return (STRICT)
// ================================

/// One extracted item tied to exact source text and global offsets.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Extraction {
    /// Category/class of extraction, e.g., "medication", "date", "clause", "character"
    pub extraction_class: String,
    /// Exact substring from the source text
    pub extraction_text: String,
    /// Inclusive char start index in the ORIGINAL full document (we shift from chunk-local)
    pub start_char: usize,
    /// Exclusive char end index
    pub end_char: usize,
    /// Optional attributes map, can hold any JSON (numbers, strings, nested objects)
    pub attributes: serde_json::Value,
}

/// A batch for one chunk of the document.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExtractionBatch {
    pub document_id: String,
    pub chunk_id: String,
    pub extractions: Vec<Extraction>,
}

// ================================
// Chunking
// ================================

/// Chunk the big document into overlapping slices for the model.
/// We use Rope for safe Unicode slicing on large strings.
fn chunk_document(text: &str, chunk_size: usize, overlap: usize) -> Vec<(String, usize, String)> {
    let rope = Rope::from_str(text);
    let len = rope.len_chars();
    let mut chunks = Vec::new();
    let mut start = 0usize;

    let pad = ((len as f64).log10().floor() as usize) + 1;
    let mut idx = 0usize;

    while start < len {
        let end = min(start + chunk_size, len);
        let slice = rope.slice(start..end).to_string();
        let chunk_id = format!("chunk-{:0width$}", idx, width = pad);
        chunks.push((slice, start, chunk_id));
        if end == len {
            break;
        }
        start = end.saturating_sub(overlap);
        idx += 1;
    }
    chunks
}

// ================================
/* OpenAI client (Structured Outputs via Responses API)
   We enforce our JSON Schema so the model MUST return schema-valid JSON.

   Response shape (simplified expected):
   {
     "output": { ... },  // may be present depending on SDK
     "content": [
        { "type":"output_text", "text":"{...JSON matching our schema...}" }
     ],
     ...
   }
*/
async fn call_openai_structured(
    client: &Client,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
    json_schema: &serde_json::Value,
    max_retries: usize,
) -> Result<serde_json::Value> {
    let url = "https://api.openai.com/v1/responses";

    // Exponential backoff parameters
    let mut attempt = 0usize;
    let mut delay_ms = 750u64;

    loop {
        attempt += 1;

        // Few-shot example to guide extraction
        let example_text = "ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.";
        let example_output = json!({
            "document_id": "demo-doc",
            "chunk_id": "example-000",
            "extractions": [
                {
                    "extraction_class": "character",
                    "extraction_text": "ROMEO",
                    "start_char": 0,
                    "end_char": 5,
                    "attributes": {"role": "speaker"}
                },
                {
                    "extraction_class": "emotion",
                    "extraction_text": "But soft!",
                    "start_char": 7,
                    "end_char": 16,
                    "attributes": {"type": "awe"}
                },
                {
                    "extraction_class": "relationship",
                    "extraction_text": "Juliet is the sun",
                    "start_char": 86,
                    "end_char": 103,
                    "attributes": {"type": "metaphor"}
                }
            ]
        }).to_string();

        let body = json!({
            "model": model,
            "input": [
              {
                "role": "system",
                "content": [{ "type":"input_text", "text": system_prompt }]
              },
              {
                "role": "user",
                "content": [{ "type":"input_text", "text": example_text }]
              },
              {
                "role": "assistant",
                "content": [{ "type":"output_text", "text": example_output }]
              },
              {
                "role": "user",
                "content": [{ "type":"input_text", "text": user_prompt }]
              }
            ],
            "text": {
              "format": {
                "type": "json_schema",
                "name": "extractions_schema",
                "schema": json_schema,
                "strict": true
              }
            }
        });

        let send_started = Instant::now();
        info!("➡️ [OpenAI] Sending request (attempt {attempt})");
        let resp = client
            .post(url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .context("OpenAI HTTP error")?;
        let send_elapsed = send_started.elapsed();

        let status = resp.status();
        let decode_started = Instant::now();
        let val: serde_json::Value = resp.json().await.context("OpenAI JSON decode")?;
        let decode_elapsed = decode_started.elapsed();

        if status.is_success() {
            info!("✅ [OpenAI] Received structured JSON (network={}ms, decode={}ms)", send_elapsed.as_millis(), decode_elapsed.as_millis());
            return Ok(val);
        }

        // If we got a rate-limit or transient error, retry
        warn!(
            "⚠️ [OpenAI] Non-success status {} on attempt {} (network={}ms, decode={}ms): {}",
            status,
            attempt,
            send_elapsed.as_millis(),
            decode_elapsed.as_millis(),
            val
        );
        if attempt >= max_retries {
            error!("❌ [OpenAI] Exhausted retries");
            return Err(anyhow::anyhow!("OpenAI error after {max_retries} attempts: {val}"));
        }

        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        delay_ms = (delay_ms as f64 * 1.75).min(5000.0) as u64;
    }
}

// ================================
// Output parsing (model → our types)
// ================================

/// Extract the JSON block that matches our schema from the Responses API payload.
fn extract_json_payload(val: &serde_json::Value) -> Result<serde_json::Value> {
    // 1) New Responses API may return structured content under `output`.
    // If `output` is an array of messages, extract first `output_text`.
    if let Some(output) = val.get("output") {
        if output.is_array() {
            if let Some(arr) = output.as_array() {
                for item in arr {
                    if let Some(contents) = item.get("content").and_then(|c| c.as_array()) {
                        for c in contents {
                            if c.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                                if let Some(text) = c.get("text").and_then(|t| t.as_str()) {
                                    let parsed: serde_json::Value = serde_json::from_str(text)
                                        .context("Failed to parse output_text in output[] as JSON")?;
                                    return Ok(parsed);
                                }
                            }
                        }
                    }
                }
            }
        } else if output.is_object() {
            // Some SDKs return the structured JSON directly under `output`
            return Ok(output.clone());
        }
    }

    // Try content[0].text
    if let Some(arr) = val.get("content").and_then(|c| c.as_array()) {
        if let Some(first) = arr.first() {
            if let Some(text) = first.get("text").and_then(|t| t.as_str()) {
                // text is JSON string; parse it
                let parsed: serde_json::Value = serde_json::from_str(text)
                    .context("Failed to parse model's output_text as JSON")?;
                return Ok(parsed);
            }
        }
    }

    // Some SDKs return `output_text` at top-level
    if let Some(text) = val.get("output_text").and_then(|t| t.as_str()) {
        let parsed: serde_json::Value =
            serde_json::from_str(text).context("Failed to parse output_text as JSON")?;
        return Ok(parsed);
    }

    Err(anyhow::anyhow!(
        "Could not find JSON matching schema in model response"
    ))
}

/// Parse into `ExtractionBatch` and shift local chunk offsets to GLOBAL offsets.
fn parse_batch_from_payload(
    payload: &serde_json::Value,
    document_id: &str,
    chunk_id: &str,
    chunk_offset: usize,
) -> Result<ExtractionBatch> {
    // Payload should match our schema (ExtractionBatch) — if model obeyed schema.
    // If it returned just an array of extractions, we handle that too.
    let mut batch: ExtractionBatch = if payload.get("extractions").is_some() {
        // Case A: full schema present
        let mut tmp: ExtractionBatch = serde_json::from_value(payload.clone())
            .context("Failed to deserialize ExtractionBatch from payload")?;
        // Overwrite doc/chunk IDs with our authoritative IDs
        tmp.document_id = document_id.to_string();
        tmp.chunk_id = chunk_id.to_string();
        tmp
    } else if payload.is_array() {
        // Case B: model returned just an array -> wrap it
        let mut exts: Vec<Extraction> = serde_json::from_value(payload.clone())
            .context("Failed to deserialize [Extraction] from payload")?;
        ExtractionBatch {
            document_id: document_id.to_string(),
            chunk_id: chunk_id.to_string(),
            extractions: exts.drain(..).collect(),
        }
    } else {
        // Case C: model returned an object like { "extractions": [...] } but missing ids
        let exts = payload
            .get("extractions")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let exts: Vec<Extraction> = exts
            .into_iter()
            .map(|e| serde_json::from_value(e).unwrap())
            .collect();
        ExtractionBatch {
            document_id: document_id.to_string(),
            chunk_id: chunk_id.to_string(),
            extractions: exts,
        }
    };

    // Shift local chunk spans -> global spans
    for e in &mut batch.extractions {
        e.start_char += chunk_offset;
        e.end_char += chunk_offset;
    }

    Ok(batch)
}

// ================================
// Parquet writer (Arrow2)
// ================================
fn write_parquet<P: AsRef<Path>>(batches: &[ExtractionBatch], path: P) -> Result<()> {
    use arrow2::array::{BinaryArray, UInt64Array, Utf8Array};
    use arrow2::chunk::Chunk;
    use arrow2::datatypes::{DataType, Field, Schema};
    use arrow2::io::parquet::write as pq;

    info!("💾 Writing {} batches to Parquet: {}", batches.len(), path.as_ref().display());

    // Flatten rows
    let mut doc_id = Vec::new();
    let mut chunk_id = Vec::new();
    let mut class = Vec::new();
    let mut text = Vec::new();
    let mut start = Vec::new();
    let mut end = Vec::new();
    let mut attrs = Vec::new();

    for b in batches {
        for e in &b.extractions {
            doc_id.push(b.document_id.clone());
            chunk_id.push(b.chunk_id.clone());
            class.push(e.extraction_class.clone());
            text.push(e.extraction_text.clone());
            start.push(e.start_char as u64);
            end.push(e.end_char as u64);
            attrs.push(serde_json::to_vec(&e.attributes).unwrap_or_default());
        }
    }

    let schema = Schema::from(vec![
        Field::new("document_id", DataType::Utf8, false),
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("extraction_class", DataType::Utf8, false),
        Field::new("extraction_text", DataType::Utf8, false),
        Field::new("start_char", DataType::UInt64, false),
        Field::new("end_char", DataType::UInt64, false),
        Field::new("attributes_json", DataType::Binary, false),
    ]);

    // Build Chunk<Box<dyn Array>> to satisfy parquet writer trait bounds
    let columns: Vec<Box<dyn arrow2::array::Array>> = vec![
        Box::new(Utf8Array::<i32>::from_slice(doc_id)),
        Box::new(Utf8Array::<i32>::from_slice(chunk_id)),
        Box::new(Utf8Array::<i32>::from_slice(class)),
        Box::new(Utf8Array::<i32>::from_slice(text)),
        Box::new(UInt64Array::from_slice(start)),
        Box::new(UInt64Array::from_slice(end)),
        Box::new(BinaryArray::<i32>::from_slice(attrs)),
    ];
    let chunk = Chunk::try_new(columns)?;

    let options = pq::WriteOptions {
        write_statistics: true,
        compression: pq::CompressionOptions::Zstd(None),
        version: pq::Version::V2,
        data_pagesize_limit: None,
    };

    // Per-column encoding (one Vec<Encoding> per leaf/column). Flat schema => one encoding per column.
    let encodings: Vec<Vec<pq::Encoding>> = schema
        .fields
        .iter()
        .map(|_| vec![pq::Encoding::Plain])
        .collect();

    let row_groups = pq::RowGroupIterator::try_new(
        std::iter::once(Ok(chunk)),
        &schema,
        options,
        encodings,
    )?;

    let mut file = File::create(path)?;
    let mut writer = pq::FileWriter::try_new(&mut file, schema, options)?;
    for rg in row_groups {
        writer.write(rg?)?;
    }
    writer.end(None)?;
    info!("✅ Parquet write complete");
    Ok(())
}

// ================================
// JSON Schema utilities
// ================================
fn enforce_no_additional_properties(schema: &mut serde_json::Value) {
    use serde_json::Value;
    match schema {
        Value::Object(map) => {
            let is_object_type = map
                .get("type")
                .and_then(|t| t.as_str())
                .map(|t| t == "object")
                .unwrap_or(false)
                || map.contains_key("properties");
            if is_object_type && !map.contains_key("additionalProperties") {
                map.insert("additionalProperties".to_string(), Value::Bool(false));
            }
            // Recurse common schema carriers
            if let Some(props) = map.get_mut("properties") {
                if let Value::Object(props_map) = props {
                    for (_k, v) in props_map.iter_mut() {
                        enforce_no_additional_properties(v);
                    }
                }
            }
            if let Some(items) = map.get_mut("items") {
                enforce_no_additional_properties(items);
            }
            for key in ["definitions", "$defs", "allOf", "anyOf", "oneOf"].iter() {
                if let Some(val) = map.get_mut(*key) {
                    match val {
                        Value::Object(obj) => {
                            for (_k, v) in obj.iter_mut() {
                                enforce_no_additional_properties(v);
                            }
                        }
                        Value::Array(arr) => {
                            for v in arr.iter_mut() {
                                enforce_no_additional_properties(v);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                enforce_no_additional_properties(v);
            }
        }
        _ => {}
    }
}

// ================================
// JSONL writer (for debugging / portability)
// ================================
fn write_jsonl<P: AsRef<Path>>(batches: &[ExtractionBatch], path: P) -> Result<()> {
    info!("🧾 Writing JSONL: {}", path.as_ref().display());
    let mut f = File::create(path)?;
    for b in batches {
        writeln!(f, "{}", serde_json::to_string(b)?)?;
    }
    Ok(())
}

// ================================
// HTML Viewer (simple, self-contained)
// ================================

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Build a very simple highlight viewer:
/// - Full source text displayed
/// - Extractions table on top
/// - Clicking a row scrolls to the highlighted region
fn write_viewer<P: AsRef<Path>>(
    full_text: &str,
    batches: &[ExtractionBatch],
    path: P,
) -> Result<()> {
    info!("🖼️ Writing HTML viewer: {}", path.as_ref().display());

    // Build a flat list of extractions with ids for anchors
    #[derive(Clone)]
    struct Row<'a> {
        id: String,
        class: &'a str,
        text: String,
        start: usize,
        end: usize,
    }

    let mut rows: Vec<Row> = Vec::new();
    for (i, b) in batches.iter().enumerate() {
        for (j, e) in b.extractions.iter().enumerate() {
            let id = format!("ext-{}-{}", i, j);
            rows.push(Row {
                id,
                class: &e.extraction_class,
                text: html_escape(&e.extraction_text),
                start: e.start_char,
                end: e.end_char,
            });
        }
    }

    // Build highlighted text by slicing and inserting <mark> tags.
    // NOTE: We need to avoid offset shifting as we insert tags; build in one pass.
    let mut pieces: Vec<String> = Vec::new();
    let mut cursor = 0usize;

    // Sort by start ascending, then end ascending (stable)
    rows.sort_by_key(|r| (r.start, r.end));

    for r in &rows {
        if r.start > full_text.len() || r.end > full_text.len() || r.start >= r.end {
            continue; // skip invalid spans silently
        }
        // text before highlight
        if r.start > cursor {
            pieces.push(html_escape(&full_text[cursor..r.start]));
        }
        // the highlighted region
        let segment = html_escape(&full_text[r.start..r.end]);
        pieces.push(format!(
            r#"<a id="{id}"></a><mark title="{class} [{start}..{end}]">{segment}</mark>"#,
            id = r.id,
            class = r.class,
            start = r.start,
            end = r.end,
            segment = segment
        ));
        cursor = r.end;
    }
    // tail
    if cursor < full_text.len() {
        pieces.push(html_escape(&full_text[cursor..]));
    }

    let highlighted = pieces.join("");

    let mut html = String::new();
    html.push_str(r#"<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>RustLangExtract Viewer</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
th, td { padding: 6px 8px; border-bottom: 1px solid #ddd; }
tr:hover { background: #f6f6f6; cursor: pointer; }
mark { background: #fffb8f; padding: 0 2px; border-radius: 2px; }
.code { white-space: pre-wrap; border: 1px solid #eee; padding: 12px; border-radius: 8px; }
.small { color: #666; font-size: 12px; }
</style>
<script>
function jumpTo(id) {
  const el = document.getElementById(id);
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // Flash the mark briefly
    el.classList.add('flash');
    setTimeout(() => el.classList.remove('flash'), 500);
  }
}
</script>
</head>
<body>
<h1>RustLangExtract Viewer</h1>
<p class="small">Click a row to jump to that exact highlighted span below.</p>
<table>
<thead><tr><th>Class</th><th>Text</th><th>Span</th></tr></thead>
<tbody>
"#);

    for r in &rows {
        html.push_str(&format!(
            r#"<tr onclick="jumpTo('{id}')"><td>{class}</td><td>{text}</td><td>[{start}..{end}]</td></tr>"#,
            id = r.id,
            class = r.class,
            text = r.text,
            start = r.start,
            end = r.end
        ));
    }

    html.push_str(r#"
</tbody>
</table>
<div class="code">"#);
    html.push_str(&highlighted);
    html.push_str(r#"</div>
</body>
</html>
"#);

    let mut f = File::create(path)?;
    f.write_all(html.as_bytes())?;
    Ok(())
}

// ================================
// Main
// ================================

#[tokio::main]
async fn main() -> Result<()> {
    // ---- Logging setup ----
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .with_target(false)
        .compact()
        .init();

    let cli = Cli::parse();

    // ---- Resolve OpenAI API key ----
    dotenvy::dotenv().ok(); // loads variables from .env if present
    let api_key = std::env::var("OPENAI_API_KEY")
        .context("Missing OPENAI_API_KEY env var. Set it before running.")?;

    // ---- Read input file ----
    let mut text = String::new();
    if cli.demo {
        info!("📄 Using built-in demo corpus (embedded)");
        text = include_str!("../example.txt").to_string();
    } else {
        let input_path = cli.input.as_ref().expect("input required unless --demo");
        info!("📄 Reading input file: {}", input_path.display());
        File::open(input_path)
            .context("Failed to open input file")?
            .read_to_string(&mut text)
            .context("Failed to read input file as UTF-8")?;
    }
    if text.trim().is_empty() {
        warn!("Input file appears empty.");
    }

    // ---- Prepare config ----
    let concurrency = cli
        .concurrency
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let cfg = Config {
        document_id: Uuid::new_v4().to_string(),
        model: cli.model,
        chunk_size: cli.chunk_size,
        overlap: cli.overlap,
        max_retries: cli.max_retries,
        concurrency,
        out_dir: cli.out_dir.clone(),
        timeout_seconds: cli.timeout_seconds.unwrap_or(120),
    };

    // ---- Prepare outputs ----
    create_dir_all(&cfg.out_dir).context("Failed to create out-dir")?;
    let parquet_path = cfg.out_dir.join("extractions.parquet");
    let jsonl_path = cfg.out_dir.join("extractions.jsonl");
    let html_path = cfg.out_dir.join("view.html");

    info!("🆔 Document ID: {}", cfg.document_id);
    info!("🧠 Model: {}", cfg.model);
    info!("⚙️  ChunkSize={}, Overlap={}, Concurrency={}", cfg.chunk_size, cfg.overlap, cfg.concurrency);

    // ---- Build JSON Schema from our Rust types ----
    // Build a strict JSON Schema manually for structured outputs
    let schema_json = serde_json::json!({
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
            "properties": {},
            "required": []
          }
                    },
                    "required": [
                        "extraction_class",
                        "extraction_text",
                        "start_char",
                        "end_char",
                        "attributes"
                    ]
                }
            }
        },
        "required": ["document_id", "chunk_id", "extractions"]
    });

    // ---- Chunk input ----
    let chunks = chunk_document(&text, cfg.chunk_size, cfg.overlap);
    info!("🪚 Created {} chunks", chunks.len());
    if chunks.is_empty() {
        warn!("No chunks created. Nothing to do.");
        return Ok(());
    }

    // ---- Progress bars ----
    let mp = MultiProgress::new();
    let pb = mp.add(ProgressBar::new(chunks.len() as u64));
    pb.set_style(
        ProgressStyle::with_template("{spinner} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message("extracting…");

    // ---- HTTP client ----
    let client = Client::builder()
        .gzip(true)
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(cfg.timeout_seconds))
        .pool_idle_timeout(Duration::from_secs(30))
        .build()
        .context("HTTP client build failed")?;

    // ---- Rayon thread pool limit ----
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.concurrency)
        .build()
        .context("Failed to build Rayon thread pool")?;

    // ---- LLM prompts ----
    let system_prompt = "You are a precise information extraction engine. \
        You ONLY return JSON that matches the provided JSON Schema. \
        For each entity, use exact substrings from the given text and provide 0-based character spans \
        relative to the given CHUNK text (start inclusive, end exclusive).";

    // Base instruction
    let mut final_instruction = String::from(
        "Task:\n\
        - Extract concrete entities across domains (e.g., people, organizations, dates, amounts, medications, clauses, emotions).\n\
        - For each entity:\n\
          - extraction_class: a short label (e.g., name, date, amount, medication, clause, emotion).\n\
          - extraction_text: EXACT substring from the text (no paraphrasing).\n\
          - start_char and end_char: 0-based indices relative to THIS CHUNK only.\n\
          - attributes: JSON with helpful fields (empty object if none).\n\
        Return JSON EXACTLY matching the provided schema. Do NOT include extra keys."
    );

    // If user provided a prompt file, override base instruction
    if let Some(prompt_path) = &cli.prompt_file {
        let mut buf = String::new();
        File::open(prompt_path)
            .context("Failed to open prompt file")?
            .read_to_string(&mut buf)
            .context("Failed to read prompt file as UTF-8")?;
        final_instruction = buf;
    }

    // Or load a built-in preset if requested (and no prompt_file override)
    if final_instruction.starts_with("Task:") {
        if let Some(preset) = &cli.preset {
            match preset.as_str() {
                "medical" => {
                    final_instruction = include_str!("../prompts/medical.txt").to_string();
                }
                "finance" => {
                    final_instruction = include_str!("../prompts/finance.txt").to_string();
                }
                "legal" => {
                    final_instruction = include_str!("../prompts/legal.txt").to_string();
                }
                _ => {}
            }
        }
    }

    // If classes are provided, append a directive to focus only on those
    if let Some(classes_csv) = &cli.classes {
        final_instruction.push_str("\n\nOnly extract the following classes (ignore others): ");
        final_instruction.push_str(classes_csv);
    }

    // ---- Process chunks in parallel ----
    let batches: Vec<ExtractionBatch> = pool.install(|| {
        chunks
            .par_iter()
            .enumerate()
            .map(|(_i, (chunk_text, offset, chunk_id))| {
                // Build user prompt with chunk context
                let user_prompt = format!(
                    "{}\n\nCHUNK_ID: {}\nCHUNK_OFFSET_IN_DOCUMENT: {}\n\nTEXT:\n{}",
                    final_instruction,
                    chunk_id,
                    offset,
                    chunk_text
                );

                // BLOCK ON async call inside rayon worker:
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio rt");
                let result = rt.block_on(call_openai_structured(
                    &client,
                    &api_key,
                    &cfg.model,
                    system_prompt,
                    &user_prompt,
                    &schema_json,
                    cfg.max_retries,
                ));

                pb.inc(1);

                match result {
                    Ok(resp) => {
                        // Pull the JSON payload matching our schema
                        match extract_json_payload(&resp)
                            .and_then(|p| parse_batch_from_payload(&p, &cfg.document_id, chunk_id, *offset))
                        {
                            Ok(mut batch) => {
                                // Optional: de-dup overlapping duplicates within this batch
                                dedupe_extractions(&mut batch.extractions);
                                Ok(batch)
                            }
                            Err(e) => {
                                error!("Parse error on {}: {}", chunk_id, e);
                                Err(e)
                            }
                        }
                    }
                    Err(e) => {
                        error!("OpenAI error on {}: {}", chunk_id, e);
                        Err(e)
                    }
                }
            })
            .filter_map(|r| r.ok())
            .collect()
    });

    pb.finish_with_message("done");

    info!("🧮 Collected {} batches", batches.len());

    // ---- Write outputs ----
    write_parquet(&batches, &parquet_path)?;
    write_jsonl(&batches, &jsonl_path)?;
    write_viewer(&text, &batches, &html_path)?;

    info!("📦 Outputs:");
    info!("  • Parquet: {}", parquet_path.display());
    info!("  • JSONL:   {}", jsonl_path.display());
    info!("  • HTML:    {}", html_path.display());

    info!("✅ All done.");
    Ok(())
}

// Simple de-duplication within a batch to reduce overlap repeats.
// Strategy: same (class, text, start, end) => keep first.
fn dedupe_extractions(extractions: &mut Vec<Extraction>) {
    use std::collections::HashSet;
    let mut seen = HashSet::<(String, String, usize, usize)>::new();
    let mut out = Vec::with_capacity(extractions.len());
    for e in extractions.drain(..) {
        let key = (
            e.extraction_class.clone(),
            e.extraction_text.clone(),
            e.start_char,
            e.end_char,
        );
        if seen.insert(key) {
            out.push(e);
        }
    }
    *extractions = out;
}
