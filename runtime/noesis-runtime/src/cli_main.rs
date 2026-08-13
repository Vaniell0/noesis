//! `noesis` — client CLI for the noesis-runtime service.
//!
//! Subcommands:
//!   run     [model] [--host HOST] [--max-tokens N]  — interactive REPL
//!   launch  <client> [args...]                       — exec client with noesis backend
//!     clients: claude, codex, ollama

mod launch;
mod run_interactive;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("run") => {
            let run_args = run_interactive::parse_args(&args[2..]);
            run_interactive::run(run_args).await
        }
        Some("launch") => {
            launch::launch(&args[2..])
        }
        _ => {
            eprintln!("noesis — client for noesis-runtime service\n");
            eprintln!("Usage:");
            eprintln!("  noesis run   [model] [--host HOST] [--max-tokens N]");
            eprintln!("  noesis launch claude|codex|ollama  [args...]");
            eprintln!();
            eprintln!("NOESIS_HOST env var overrides default http://127.0.0.1:11435");
            std::process::exit(1);
        }
    }
}
