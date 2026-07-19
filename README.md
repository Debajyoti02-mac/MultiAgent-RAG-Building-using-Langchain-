# Multi-Agent RAG System (Manual, Pre-LangGraph)

A hand-rolled multi-agent router built with LangChain, before moving to LangGraph abstractions. Routes user queries to one of three tools: a calculator, a local document retriever (ChromaDB), or a web search fallback.

## Architecture

```
User Query
    │
    ▼
 fetch() — classifier
    │
    ├── "cal"     → calculator tool (arithmetic only)
    ├── "retrive" → retrival tool (ChromaDB local search)
    │                   │
    │                   └── if no relevant chunks found → web_search tool (DuckDuckGo)
    └── "General"  → fallback: raw LLM call (no tools)
```

**Design intent:** every query should hit `retrive` first unless it's pure arithmetic. `web_search` is only meant to be reached as a fallback when local retrieval comes up empty — not as a top-level route.

## Stack

- **LLM:** ChatGroq, `llama-3.1-8b-instant`
- **Vector store:** ChromaDB (`PersistentClient`, local path `./Database_VD`)
- **Embeddings:** `DefaultEmbeddingFunction` (Chroma's built-in ONNX model — switched from `SentenceTransformerEmbeddingFunction`/`all-MiniLM-L6-v2` due to HuggingFace download hangs)
- **Document loading:** `PyPDFLoader`
- **Chunking:** `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=140)
- **Web search:** `DuckDuckGoSearchRun`
- **Env management:** `dotenv` (`GROQ_API_KEY`)

## Components

### `calculator` tool
Evaluates arithmetic expressions. **Known issue:** currently uses `eval()`, which is unsafe for anything beyond trusted numeric input. Should be swapped to `ast.literal_eval` or a proper expression parser before this goes near untrusted input.

### `retrival` tool
Queries the Chroma collection, filters results by L2 distance threshold (`< 1.0`), and returns concatenated matching chunks, or a "not relevant" signal if nothing clears the threshold.

### `web_search` tool
Wraps `DuckDuckGoSearchRun`. Intended fallback when `retrival` finds nothing relevant.

### `fetch()` — the router
Originally an LLM-based classifier prompted to output one of `cal` / `retrive` / `web`. This was the source of most bugs in this project (see below) — the model would narrate its reasoning in full sentences instead of returning a single clean label, which meant the classifier silently always fell through to a no-tools `'General'` fallback.

**Current direction:** moving arithmetic detection to a deterministic regex check instead of relying on the LLM to self-report, since `cal` vs `retrive` is a solved problem that doesn't need a model call:

```python
import re

def fetch(query: str):
    if re.fullmatch(r'[\d\s\+\-\*/\.\(\)]+', query.strip()):
        return "cal"
    return "retrive"
```

## Known bugs fixed during development

1. **Classifier never matched valid labels** — prompt let the LLM narrate instead of emitting a bare word, so `label in AGENTS` was always `False` and every query silently fell to a plain, tool-less LLM call. This produced convincing-looking but fake "tool call" narration text and stale/hallucinated answers (e.g. outdated president info, invented statistics).
2. **`retrival` early-return bug** — the relevance check and return statement were nested inside the `for` loop, so it only ever evaluated the first search result instead of all five.
3. **Return value order mismatch** — `run()` returns `(state, answer)` but was being unpacked as `answer, state = run(...)`.
4. **`Exception` used as a parameter name** in `calculator`, shadowing the Python builtin.
5. **Embedding hang** — `SentenceTransformerEmbeddingFunction` stalled on first-run HuggingFace model download with no visible error. Switched to Chroma's `DefaultEmbeddingFunction` to avoid the dependency on HF connectivity.
6. **Debug print pollution** — a leftover `print(response.content)` inside the classifier was printing raw model narration to stdout, which is what made the routing bug visible in the first place.

## Open items

- [ ] Finish swapping `fetch()` to deterministic regex-based routing (remove LLM call from the routing step entirely)
- [ ] Replace `eval()` in `calculator` with `ast.literal_eval` or a safe expression evaluator
- [ ] Confirm `web_search` is only ever reached as a fallback from `retrival`, not as a top-level route
- [ ] Add batching to `collection.add()` for larger document sets (Chroma has an internal batch size limit, ~5461 by default)
- [ ] Use content-hash IDs instead of positional `str(i)` IDs to make re-ingestion idempotent and safe for incremental loads

## Next phase

Once the manual routing is stable and verified, migrate this pipeline to LangGraph for explicit state-graph control over the retrieve → web_search fallback path.# MultiAgent-RAG-Building-using-Langchain-
