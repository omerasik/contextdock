import asyncio
import hmac
import json
import logging
import os
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .evaluate import evaluate
from .retrieval import Encoder, LocalEncoder, RemoteEncoder, chunks, excerpts, rank
from .store import Conflict, Store

log = logging.getLogger(__name__)
DEMO_TOKENS = {"alice": "contextdock-demo-alice", "bob": "contextdock-demo-bob"}
Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,39}$")]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DocumentInput(Input):
    collection: Slug
    title: Annotated[str, Field(min_length=1, max_length=120)]
    text: Annotated[str, Field(min_length=1, max_length=18_000)]


class SearchInput(Input):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    collection: Slug | None = None
    limit: Annotated[int, Field(ge=1, le=10)] = 5
    threshold: Annotated[float, Field(ge=0.1, le=1)] = 0.2


class MemoryInput(Input):
    value: Annotated[str, Field(min_length=1, max_length=2000)] | None
    expected: UUID | None = None


class Retention(Input):
    retention_days: Annotated[int, Field(ge=1, le=365)]


def create_app(path: Path, tokens: dict[str, str], encoder: Encoder, *, demo: bool = False) -> FastAPI:
    if not tokens or len(tokens) > 20 or len(set(tokens.values())) != len(tokens) or any(
        not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", name) or len(token) < (1 if demo else 32) or len(token) > 160
        for name, token in tokens.items()
    ):
        raise ValueError("Configure distinct tenant tokens with at least 32 characters for deployment.")
    store = Store(path, encoder.identity)

    async def cleanup():
        while True:
            for name in tokens:
                try:
                    await asyncio.to_thread(store.cleanup, name)
                except Exception as error:
                    log.error("retention_cleanup_failed kind=%s", type(error).__name__)
            await asyncio.sleep(300)

    @asynccontextmanager
    async def lifespan(_app):
        cleaner = asyncio.create_task(cleanup())
        try:
            yield
        finally:
            cleaner.cancel()
            with suppress(asyncio.CancelledError):
                await cleaner

    app = FastAPI(title="ContextDock", lifespan=lifespan)

    def tenant(authorization: Annotated[str | None, Header()] = None) -> str:
        supplied = (authorization or "").encode()
        for name, token in tokens.items():
            if hmac.compare_digest(supplied, f"Bearer {token}".encode()):
                return name
        raise HTTPException(401, "tenant_token_required")

    Scope = Annotated[str, Depends(tenant)]
    api = APIRouter(prefix="/api")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/config")
    def config():
        return {"demo": demo and tokens == DEMO_TOKENS}

    def encode(texts: list[str]):
        try:
            return encoder.encode(texts)
        except RuntimeError as error:
            raise HTTPException(503, "embedding_provider_unavailable") from error
        except ValueError as error:
            raise HTTPException(422, "no_searchable_terms") from error

    @api.get("/settings")
    def settings(owner: Scope):
        return {"tenant": owner, "encoder": "local" if isinstance(encoder, LocalEncoder) else "configured", **store.settings(owner)}

    @api.put("/settings")
    def update_settings(owner: Scope, body: Retention):
        return store.settings(owner, body.retention_days)

    @api.get("/documents")
    def documents(owner: Scope):
        return store.documents(owner)

    @api.post("/documents", status_code=201)
    def upload(owner: Scope, body: DocumentInput):
        parts = list(chunks(body.text))
        return store.ingest(owner, body.collection, body.title, body.text, parts, encode([part[2] for part in parts]))

    @api.get("/documents/{document_id}")
    def source(owner: Scope, document_id: UUID):
        document = store.source(owner, str(document_id))
        if document is None:
            raise HTTPException(404, "document_not_found")
        return document

    @api.delete("/documents/{document_id}")
    def delete(owner: Scope, document_id: UUID):
        if not store.delete(owner, str(document_id)):
            raise HTTPException(404, "document_not_found")
        return {"deleted": True}

    @api.post("/search")
    def search(owner: Scope, body: SearchInput):
        embedding = encode([body.query])[0]
        hits = store.search(owner, body.collection, embedding, min(body.limit * 3, 30))
        sources = rank(body.query, hits, body.limit, body.threshold)
        return {"sources": sources, "answer": excerpts(body.query, sources)}

    @api.get("/memory")
    def memories(owner: Scope):
        return store.memories(owner)

    @api.put("/memory/{name}")
    def remember(owner: Scope, name: Slug, body: MemoryInput):
        try:
            result = store.remember(owner, name, str(body.expected) if body.expected else None, body.value)
        except Conflict as error:
            raise HTTPException(409, str(error)) from error
        if result is None:
            raise HTTPException(404, "memory_not_found")
        return result

    @api.get("/audit")
    def audit(owner: Scope):
        return store.audit_log(owner)

    @api.get("/evaluation")
    def evaluation(_owner: Scope):
        if not demo:
            raise HTTPException(404, "evaluation_available_in_demo")
        return evaluate()

    @app.middleware("http")
    async def security(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if not request.url.path.startswith("/api/"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    app.include_router(api)
    app.mount("/", StaticFiles(directory=Path(__file__).with_name("static"), html=True), name="dashboard")
    return app


def from_env():
    demo = os.getenv("DEMO_MODE") == "true"
    tokens = json.loads(os.getenv("TENANT_TOKENS_JSON", "{}")) or (DEMO_TOKENS if demo else {})
    if not isinstance(tokens, dict) or any(not isinstance(value, str) for value in tokens.values()):
        raise ValueError("TENANT_TOKENS_JSON must map tenant names to bearer tokens.")
    encoder: Encoder = LocalEncoder()
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    if provider == "openai-compatible":
        endpoint = os.getenv("EMBEDDING_ENDPOINT", "https://api.openai.com/v1/embeddings")
        model, key = os.getenv("EMBEDDING_MODEL"), os.getenv("EMBEDDING_API_KEY")
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment or not model or not key:
            raise ValueError("Configure an HTTPS embedding endpoint, model and credential.")
        encoder = RemoteEncoder(endpoint, model, key)
    elif provider != "local":
        raise ValueError("Unknown embedding configuration.")
    return create_app(Path(os.getenv("DATABASE_PATH", "data/context.sqlite")), tokens, encoder, demo=demo)
