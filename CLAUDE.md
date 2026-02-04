# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Always use `uv` to run Python commands and manage dependencies in this project. Never use `python`, `pip`, or `pip install` directly. Use `uv add <package>` to add dependencies and `uv remove <package>` to remove them.

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh

# Run manually (from project root)
cd backend && uv run uvicorn app:app --reload --port 8000

# Run a single backend module directly
cd backend && uv run python <module>.py
```

The server runs at `http://localhost:8000` with API docs at `/docs`.

There are no tests, linter, or formatter configured in this project.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. It has a Python/FastAPI backend and a vanilla JavaScript frontend.

### Query Flow

1. **Frontend** (`frontend/script.js`) sends `POST /api/query` with `{query, session_id}`
2. **API layer** (`backend/app.py`) validates via Pydantic, delegates to `RAGSystem.query()`
3. **RAG orchestrator** (`backend/rag_system.py`) wraps the query, retrieves conversation history from `SessionManager`, and calls `AIGenerator` with tool definitions
4. **AI generator** (`backend/ai_generator.py`) calls the Claude API with the `search_course_content` tool available. Claude decides whether to search or answer directly.
5. **If Claude invokes the tool**: `ToolManager` dispatches to `CourseSearchTool` (`backend/search_tools.py`), which calls `VectorStore.search()`. Results are sent back to Claude in a second API call for synthesis.
6. **Vector store** (`backend/vector_store.py`) uses ChromaDB with SentenceTransformer embeddings (`all-MiniLM-L6-v2`). Two collections: `course_catalog` (course metadata for fuzzy name resolution) and `course_content` (chunked text for semantic search).
7. Response flows back with sources extracted from `ToolManager.get_last_sources()`.

### Key Design Decisions

- **Tool-based search**: Claude decides when to search via Anthropic tool calling (`tool_choice: auto`), rather than always searching. General knowledge questions are answered directly.
- **Two-collection pattern**: `course_catalog` enables fuzzy course name matching (e.g., "MCP" resolves to full title) before filtering `course_content`.
- **In-memory sessions**: `SessionManager` stores conversation history in a dict, keyed by session ID. History is capped at `MAX_HISTORY` exchanges (default 2) and appended to the system prompt as plain text.
- **Startup document loading**: On server start, `app.py:startup_event` loads all `.txt`/`.pdf`/`.docx` files from `docs/` into ChromaDB, skipping already-indexed courses by title.

### Course Document Format

Files in `docs/` follow a structured format parsed by `DocumentProcessor`:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
[content...]
```

Text is chunked at 800 characters with 100-character overlap (configurable in `backend/config.py`).

### Configuration

All settings are in `backend/config.py` as a dataclass. The `ANTHROPIC_API_KEY` is loaded from a `.env` file in the project root. The ChromaDB database is persisted at `backend/chroma_db/`.
