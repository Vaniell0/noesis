//! `noesis-runtime run [model] [--host HOST]` — interactive chat REPL.
//!
//! Connects to a running noesis-runtime HTTP shim and provides a
//! readline-style conversation loop over `/v1/messages` (Anthropic SSE).
//! History is kept client-side; each turn sends the full accumulated list
//! so the model has full context (standard transformer-style approach until
//! lens-based stateful sessions are wired up).

use std::io::{self, BufRead, Write};

use anyhow::Result;
use reqwest::header;
use serde_json::{json, Value};
use tokio_stream::StreamExt;

const DEFAULT_HOST: &str = "http://127.0.0.1:11435";
const DEFAULT_MAX_TOKENS: usize = 512;

pub fn parse_args(args: &[String]) -> RunArgs {
    let mut host = std::env::var("NOESIS_HOST")
        .unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let mut model = None;
    let mut max_tokens = DEFAULT_MAX_TOKENS;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" | "-H" => {
                if let Some(h) = args.get(i + 1) {
                    host = h.clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--max-tokens" | "-n" => {
                if let Some(n) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    max_tokens = n;
                }
                i += 2;
            }
            arg if !arg.starts_with('-') && model.is_none() => {
                model = Some(arg.to_string());
                i += 1;
            }
            _ => { i += 1; }
        }
    }
    RunArgs { host, model, max_tokens }
}

pub struct RunArgs {
    pub host: String,
    pub model: Option<String>,
    pub max_tokens: usize,
}

pub async fn run(args: RunArgs) -> Result<()> {
    let client = reqwest::Client::new();

    // Discover model name from /api/tags if not provided.
    let model = match args.model {
        Some(m) => m,
        None => discover_model(&client, &args.host).await?,
    };

    eprintln!("noesis  {}  @ {}", model, args.host);
    eprintln!("Ctrl-D or /bye to exit, /reset to clear history");
    eprintln!();

    let mut messages: Vec<Value> = Vec::new();

    let stdin = io::stdin();
    loop {
        print!(">>> ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/bye" || trimmed == "/exit" || trimmed == "/quit" {
            break;
        }
        if trimmed == "/reset" {
            messages.clear();
            eprintln!("(history cleared)");
            continue;
        }

        messages.push(json!({"role": "user", "content": trimmed}));

        let response_text = stream_response(
            &client,
            &args.host,
            &model,
            &messages,
            args.max_tokens,
        ).await;

        println!();
        println!();

        match response_text {
            Ok(text) => {
                if !text.is_empty() {
                    messages.push(json!({"role": "assistant", "content": text}));
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                messages.pop();
            }
        }
    }

    Ok(())
}

async fn discover_model(client: &reqwest::Client, host: &str) -> Result<String> {
    let resp = client
        .get(format!("{host}/api/tags"))
        .send()
        .await?
        .json::<Value>()
        .await?;
    resp["models"][0]["name"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("no models returned from {host}/api/tags"))
}

/// POST to `/v1/messages` with SSE streaming. Prints tokens as they arrive
/// and returns the full assembled response text.
async fn stream_response(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    messages: &[Value],
    max_tokens: usize,
) -> Result<String> {
    let resp = client
        .post(format!("{host}/v1/messages"))
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-api-key", "noesis-local")
        .json(&json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": true,
        }))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await?;
        anyhow::bail!("{status}: {body}");
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut assembled = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE events (each ends with \n\n).
        while let Some(pos) = buf.find("\n\n") {
            let event_str = buf[..pos].to_string();
            buf = buf[pos + 2..].to_string();

            // Extract the `data:` line from the event.
            for line in event_str.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        if v["type"] == "content_block_delta" {
                            if let Some(text) = v["delta"]["text"].as_str() {
                                print!("{text}");
                                io::stdout().flush().ok();
                                assembled.push_str(text);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(assembled)
}
