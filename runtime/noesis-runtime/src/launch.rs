//! `noesis-runtime launch <client> [args...]`
//!
//! Sets the correct environment variables so the named client routes its
//! API calls through the local noesis-runtime HTTP shim, then execs into it.
//!
//! Supported clients:
//!   claude   — sets ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY, execs `claude`
//!   codex    — sets OPENAI_BASE_URL + OPENAI_API_KEY, execs `codex`
//!   ollama   — sets OLLAMA_HOST, execs `ollama run <model>`
//!
//! The host is read from NOESIS_HOST env var (default: http://127.0.0.1:11435).
//! Any extra args after the client name are forwarded verbatim.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::os::unix::process::CommandExt;

use anyhow::{bail, Result};

const DEFAULT_HOST: &str = "http://127.0.0.1:11435";

pub fn launch(args: &[String]) -> Result<()> {
    let client = match args.first() {
        Some(c) => c.as_str(),
        None => bail!("Usage: noesis-runtime launch <client> [args...]\n\
                       Clients: claude, codex, ollama"),
    };
    let extra = &args[1..];
    let host = std::env::var("NOESIS_HOST")
        .unwrap_or_else(|_| DEFAULT_HOST.to_string());
    // API key sent to clients. Our server accepts any value; the key is only
    // forwarded in request headers and never validated server-side.
    // Override with NOESIS_API_KEY env var or set in your shell profile.
    let api_key = std::env::var("NOESIS_API_KEY")
        .unwrap_or_else(|_| "noesis".to_string());

    match client {
        "claude" => {
            let mut cmd = std::process::Command::new("claude");
            cmd.env("ANTHROPIC_BASE_URL", &host)
               .env("ANTHROPIC_API_KEY", &api_key);
            cmd.args(extra);
            let err = cmd.exec(); // replaces the process image on success
            bail!("exec claude: {err}");
        }
        "codex" => {
            let mut cmd = std::process::Command::new("codex");
            cmd.env("OPENAI_BASE_URL", format!("{host}/v1"))
               .env("OPENAI_API_KEY", &api_key);
            cmd.args(extra);
            let err = cmd.exec();
            bail!("exec codex: {err}");
        }
        "ollama" => {
            let model = discover_model(&host)
                .unwrap_or_else(|_| "noesis".to_string());
            let mut cmd = std::process::Command::new("ollama");
            cmd.env("OLLAMA_HOST", &host);
            if extra.is_empty() {
                cmd.args(["run", &model]);
            } else {
                cmd.args(extra);
            }
            let err = cmd.exec();
            bail!("exec ollama: {err}");
        }
        other => bail!("unknown client '{other}'. Supported: claude, codex, ollama"),
    }
}

/// Minimal blocking GET over HTTP/1.0 — no deps, no async.
fn http_get(url: &str) -> Result<String> {
    let without_scheme = url.trim_start_matches("http://");
    let (addr, path) = if let Some(slash) = without_scheme.find('/') {
        (&without_scheme[..slash], &without_scheme[slash..])
    } else {
        (without_scheme, "/")
    };

    let mut stream = TcpStream::connect(addr)?;
    write!(stream, "GET {path} HTTP/1.0\r\nHost: {addr}\r\nConnection: close\r\n\r\n")?;

    let mut reader = BufReader::new(stream);
    // Skip response headers.
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() { break; }
    }
    let mut body = String::new();
    reader.read_to_string(&mut body)?;
    Ok(body)
}

fn discover_model(host: &str) -> Result<String> {
    let body = http_get(&format!("{host}/api/tags"))?;
    let v: serde_json::Value = serde_json::from_str(&body)?;
    v["models"][0]["name"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("no model in /api/tags"))
}
