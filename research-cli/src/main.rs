/// Research Assistant CLI
/// Phase 4 – Rust frontend that talks to the FastAPI backend.
///
/// Usage:
///   research-cli ask "What does the paper say about transformers?"
///   research-cli ask --stream "Summarise the key findings"
///   research-cli ask --top-k 8 "Explain the methodology"

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const DEFAULT_API_URL: &str = "http://localhost:8000";

// ── CLI structure ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "research-cli",
    about = "Research assistant powered by LangGraph + Mistral",
    version
)]
struct Cli {
    /// Base URL of the FastAPI backend
    #[arg(long, env = "RESEARCH_API_URL", default_value = DEFAULT_API_URL)]
    api_url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ask a question against the knowledge base
    Ask {
        /// Your question
        query: String,

        /// Stream the response token by token (SSE)
        #[arg(long, short = 's')]
        stream: bool,

        /// Number of context chunks to retrieve
        #[arg(long, default_value_t = 5)]
        top_k: u32,
    },
}

// ── API types ─────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct AskRequest {
    query: String,
    top_k: u32,
}

#[derive(Deserialize)]
struct AskResponse {
    answer: String,
    sources: Vec<String>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn print_header() {
    println!("{}", "─".repeat(60).dimmed());
    println!("{}", " Research Assistant".bold().cyan());
    println!("{}", "─".repeat(60).dimmed());
}

fn print_sources(sources: &[String]) {
    if sources.is_empty() {
        return;
    }
    println!("\n{}", "Sources:".bold().yellow());
    for s in sources {
        println!("  {} {}", "•".yellow(), s.italic());
    }
}

// ── Blocking ask ──────────────────────────────────────────────────────────────

async fn ask_blocking(client: &Client, api_url: &str, req: &AskRequest) -> Result<()> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    spinner.set_message("Searching knowledge base…");
    spinner.enable_steady_tick(Duration::from_millis(80));

    let url = format!("{}/ask", api_url);
    let resp = client
        .post(&url)
        .json(req)
        .send()
        .await
        .context("Failed to reach the API")?;

    spinner.finish_and_clear();

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("API error {}: {}", status, body);
    }

    let data: AskResponse = resp.json().await.context("Failed to parse API response")?;

    println!("\n{}", "Answer:".bold().green());
    println!("{}", data.answer);
    print_sources(&data.sources);

    Ok(())
}

// ── Streaming ask (SSE) ───────────────────────────────────────────────────────

async fn ask_stream(client: &Client, api_url: &str, req: &AskRequest) -> Result<()> {
    let url = format!("{}/ask/stream", api_url);

    let resp = client
        .post(&url)
        .json(req)
        .send()
        .await
        .context("Failed to reach the API")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("API error {}: {}", status, body);
    }

    println!("\n{}", "Answer:".bold().green());

    let mut stream = resp.bytes_stream();
    let mut sources: Vec<String> = vec![];
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("Stream error")?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        // SSE lines end with \n\n; process complete events
        while let Some(pos) = buffer.find("\n\n") {
            let event_block = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();

            let mut event_type = String::new();
            let mut data = String::new();

            for line in event_block.lines() {
                if let Some(stripped) = line.strip_prefix("event: ") {
                    event_type = stripped.to_string();
                } else if let Some(stripped) = line.strip_prefix("data: ") {
                    data = stripped.to_string();
                }
            }

            match event_type.as_str() {
                "sources" => {
                    if let Ok(parsed) = serde_json::from_str::<Vec<String>>(&data) {
                        sources = parsed;
                    }
                }
                "token" => {
                    // Unescape JSON string value (data is a raw string)
                    let token = data.trim_matches('"').replace("\\n", "\n");
                    print!("{}", token);
                }
                "done" => {
                    println!(); // newline after streamed answer
                }
                _ => {}
            }
        }
    }

    print_sources(&sources);
    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env if present (for RESEARCH_API_URL etc.)
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("Failed to build HTTP client")?;

    print_header();

    match cli.command {
        Commands::Ask { query, stream, top_k } => {
            println!("{} {}", "Query:".bold(), query.italic());

            let req = AskRequest {
                query: query.clone(),
                top_k,
            };

            if stream {
                ask_stream(&client, &cli.api_url, &req).await?;
            } else {
                ask_blocking(&client, &cli.api_url, &req).await?;
            }
        }
    }

    println!("{}", "─".repeat(60).dimmed());
    Ok(())
}