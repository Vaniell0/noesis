//! `noesis-runtime run [model] [--host HOST]` — interactive REPL with persistent state.
//!
//! Uses the lens system to maintain WKV state across turns so only new tokens
//! are processed each round — never re-reads the full conversation history.
//!
//! Turn protocol over `/api/generate`:
//!   Turn 1: full ChatML prompt  →  save state to lens
//!   Turn N: only the new user turn suffix  →  load lens, append, save back
//!
//! This is the RWKV-native approach: the model carries context in WKV state,
//! not in the prompt string.

use std::io::{self, BufRead, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use reqwest::header;
use serde_json::{json, Value};
use tokio_stream::StreamExt;

const DEFAULT_HOST: &str = "http://127.0.0.1:11435";
const DEFAULT_MAX_TOKENS: usize = 512;

/// Stop strings for ChatML format.
const STOPS: &[&str] = &["<|im_end|>", "<|endoftext|>", "\n<|im_start|>"];

pub fn parse_args(args: &[String]) -> RunArgs {
    let mut host = std::env::var("NOESIS_HOST")
        .unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let mut model = None;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut system: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" | "-H" => {
                if let Some(h) = args.get(i + 1) { host = h.clone(); i += 2; } else { i += 1; }
            }
            "--max-tokens" | "-n" => {
                if let Some(n) = args.get(i + 1).and_then(|s| s.parse().ok()) { max_tokens = n; }
                i += 2;
            }
            "--system" | "-s" => {
                if let Some(s) = args.get(i + 1) { system = Some(s.clone()); i += 2; } else { i += 1; }
            }
            arg if !arg.starts_with('-') && model.is_none() => {
                model = Some(arg.to_string());
                i += 1;
            }
            _ => { i += 1; }
        }
    }
    RunArgs { host, model, max_tokens, system }
}

pub struct RunArgs {
    pub host: String,
    pub model: Option<String>,
    pub max_tokens: usize,
    pub system: Option<String>,
}

pub async fn run(args: RunArgs) -> Result<()> {
    let client = reqwest::Client::new();

    let model = match args.model {
        Some(m) => m,
        None => discover_model(&client, &args.host).await?,
    };

    let lens_id = format!("run-{}", SystemTime::now()
        .duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0));

    eprintln!("noesis  {}  @ {}  [lens: {}]", model, args.host, lens_id);
    eprintln!("Ctrl-D or /bye to exit, /reset for new session");
    eprintln!();

    let system_prompt = args.system.unwrap_or_else(|| {
        "You are noesis, a persistent cognitive runtime. Answer concisely.".into()
    });

    let stdin = io::stdin();
    let mut first_turn = true;
    let mut current_lens = lens_id.clone();

    loop {
        print!(">>> ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        if trimmed == "/bye" || trimmed == "/exit" || trimmed == "/quit" { break; }
        if trimmed == "/reset" {
            // New lens = new session; old lens stays on disk for reference.
            current_lens = format!("run-{}", SystemTime::now()
                .duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0));
            first_turn = true;
            eprintln!("(new session: {})", current_lens);
            continue;
        }

        // Build the prompt for this turn.
        let prompt = if first_turn {
            // Full ChatML: system + first user turn.
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                system_prompt, trimmed
            )
        } else {
            // Continuation: state already has previous turns via lens.
            // The state ends after the previous <|im_end|> (assistant close).
            // Append the new user turn.
            format!(
                "\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                trimmed
            )
        };

        let ok = stream_generate(
            &client,
            &args.host,
            &model,
            &prompt,
            &current_lens,
            args.max_tokens,
        ).await?;

        println!();
        println!();

        if ok {
            first_turn = false;
        }
    }

    Ok(())
}

async fn discover_model(client: &reqwest::Client, host: &str) -> Result<String> {
    let resp = client.get(format!("{host}/api/tags"))
        .send().await?.json::<Value>().await?;
    resp["models"][0]["name"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("no models from {host}/api/tags"))
}

/// POST to `/api/generate` with lens_id + NDJSON streaming.
/// Prints tokens as they arrive. Returns true if generation succeeded.
async fn stream_generate(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    prompt: &str,
    lens_id: &str,
    max_tokens: usize,
) -> Result<bool> {
    let body = json!({
        "model": model,
        "prompt": prompt,
        "lens_id": lens_id,
        "stream": true,
        "options": { "num_predict": max_tokens },
        "stop": STOPS,
    });

    let resp = client
        .post(format!("{host}/api/generate"))
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_txt = resp.text().await?;
        anyhow::bail!("{status}: {body_txt}");
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut ok = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        // Each NDJSON line is a complete JSON object.
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim().to_string();
            buf = buf[pos + 1..].to_string();

            if line.is_empty() { continue; }
            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                if let Some(delta) = v["response"].as_str() {
                    if !delta.is_empty() {
                        print!("{delta}");
                        io::stdout().flush().ok();
                    }
                }
                if v["done"].as_bool() == Some(true) {
                    ok = true;
                }
            }
        }
    }

    Ok(ok)
}
