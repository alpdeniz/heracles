# Heracles

Search a folder by meaning, not just keywords. Unix only (macOS / Linux).

## Install

Needs [Rust](https://rustup.rs)

```sh
make install   # builds heracles into ~/.cargo/bin and runs heracles init
```

`make uninstall` removes the binary; delete `~/.heracles/` yourself to wipe data.

## Usage

```sh
heracles index ~/notes
heracles search "how did we handle retries"          # hybrid (default)
heracles search "idempotency" --mode grep            # ripgrep only
heracles search "retries" --mode semantic            # embeddings only
heracles search "retries" -p ~/notes                 # scope; repeat -p
heracles search "setup" --include '*.md'
heracles index . --exclude node_modules --exclude '*.lock'
heracles index ~/notes --depth 1                     # 0 = unlimited (default)
heracles list
heracles config
```

`heracles --help` and `heracles <command> --help` list every flag.

Search auto-indexes the current directory (or `-p` folders) unless you use `--mode grep`. Results default to 10 (`-n`). Ranking uses all indexed files unless `-p` narrows it; `--depth` only limits the walk (and ripgrep), not which indexed files can rank.

### Filters

`--include` / `--exclude` take names or globs (`.gitignore` syntax), matched at any depth. Quote globs. A leading `/` is an absolute path.

On `search` and `list`, they narrow what is shown. On `index`, they limit what is (re-)embedded or forgotten; other indexed files stay. A folder indexed with `--include` or a limited `--depth` is not marked fully covered, so a later search there indexes the rest. Unlimited `index --exclude` does count as covered.

### Search modes

| Mode | Behavior |
|------|----------|
| `hybrid` (default) | Grep first, then refresh index; score = `(1 − w) × semantic + w × grep` |
| `semantic` | Embedding similarity over indexed chunks |
| `grep` | In-process ripgrep on the live tree; does not touch vector memory |

`w` is `search.grep_weight` (default `0.35`). Grep does not see PDF text; hybrid can via embeddings.

## Config and data

Everything lives under `~/.heracles/` (override with `HERACLES_HOME`):

```
~/.heracles/
├── config.toml
└── memory/          # catalog + int8 vectors; flocked so one process at a time
```

Default `config.toml`:

```toml
[index]
max_file_size_kb = 1024
max_pdf_size_kb = 51200
respect_gitignore = false
chunk_tokens = 1024
chunk_overlap_tokens = 64
read_workers = 4

[search]
grep_weight = 0.35

[embedding]
url = "http://localhost:8080/v1/embeddings"
model = "Qwen/Qwen3-embedding-0.6B"
# store and compare 512 only - e.g. of 1024 vector elements from Qwen3-0.6B
dimensions = 512
concurrent_requests = 2
timeout_secs = 60
query_prefix = """
Instruct: Given a search query, retrieve relevant passages that answer the query
Query:"""
```

Index walks include hidden files, skip binary/non-UTF-8 and oversized files, and honor `.ignore`. `.gitignore` applies only when `respect_gitignore = true`. PDFs become Markdown via [`pdf_oxide`](https://crates.io/crates/pdf_oxide) (up to `max_pdf_size_kb`); scanned PDFs with no text layer yield nothing.

Re-index is incremental (mtime/size); embedding failures keep progress and resume next time. Changing `embedding.model`, `embedding.dimensions`, or chunk settings requires clearing `memory/` and re-indexing.

`dimensions` truncates longer server vectors (Matryoshka-friendly). `query_prefix` is prepended to search queries only — set `""` if the model does not use instructions. `read_workers` is file read/chunk concurrency; embedding uses `concurrent_requests`.

### Embedding server

Any OpenAI-compatible `/v1/embeddings` endpoint (e.g. llama.cpp `llama-server`). If the server uses `--api-key`:

```toml
[embedding]
api_key = "..."
```

or `export HERACLES_EMBEDDING_API_KEY=...` (env wins).

## Development

```sh
make            # list targets
make check      # fmt + clippy + tests
make test
make run ARGS="search 'query'"
cargo test --test semantic_search -- --include-ignored   # needs a live embedding server
```

## License

[MIT](LICENSE)
