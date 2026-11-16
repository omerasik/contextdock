import hashlib
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import sqlite_vec

from .retrieval import DIMENSIONS


class Conflict(Exception):
    pass


class Store:
    def __init__(self, path: Path, provider: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript(f"""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS index_config (id INTEGER PRIMARY KEY CHECK(id=1), provider TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, collection TEXT NOT NULL, title TEXT NOT NULL,
                    digest TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL,
                    UNIQUE(tenant, collection, digest)
                );
                CREATE INDEX IF NOT EXISTS document_scope ON documents(tenant, collection, created);
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                    embedding float[{DIMENSIONS}] distance_metric=cosine, tenant text partition key, collection text partition key
                );
                CREATE TABLE IF NOT EXISTS memory (
                    tenant TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, version TEXT NOT NULL,
                    PRIMARY KEY(tenant, name)
                );
                CREATE TABLE IF NOT EXISTS settings (tenant TEXT PRIMARY KEY, retention_days INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY, tenant TEXT NOT NULL, action TEXT NOT NULL, object_id TEXT, at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS audit_tenant ON audit(tenant, id DESC);
            """)
            db.execute("INSERT OR IGNORE INTO index_config VALUES(1, ?)", (provider,))
            if db.execute("SELECT provider FROM index_config WHERE id=1").fetchone()[0] != provider:
                raise ValueError("The database was indexed by another embedding configuration.")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            db.execute("PRAGMA foreign_keys=ON")
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def audit(db, tenant, action, object_id=None):
        db.execute("INSERT INTO audit(tenant,action,object_id,at) VALUES(?,?,?,?)", (tenant, action, object_id, time.time()))

    @staticmethod
    def retention(db, tenant):
        row = db.execute("SELECT retention_days FROM settings WHERE tenant=?", (tenant,)).fetchone()
        return row[0] if row else 30

    def settings(self, tenant, days=None):
        with self.connect() as db:
            if days is not None:
                db.execute("INSERT INTO settings VALUES(?,?) ON CONFLICT(tenant) DO UPDATE SET retention_days=excluded.retention_days",
                           (tenant, days))
                self.audit(db, tenant, "retention_changed")
            return {"retention_days": self.retention(db, tenant)}

    def ingest(self, tenant, collection, title, text, parts, embeddings):
        digest = hashlib.sha256(text.encode()).hexdigest()
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT id FROM documents WHERE tenant=? AND collection=? AND digest=? AND expires>?",
                                  (tenant, collection, digest, now)).fetchone()
            if existing:
                return {"id": existing["id"], "duplicate": True}
            self.purge(db, tenant)
            doc_id = str(uuid4())
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)",
                       (doc_id, tenant, collection, title, digest, text, now, now + self.retention(db, tenant) * 86_400))
            for (start, end, part), embedding in zip(parts, embeddings, strict=True):
                row = db.execute("INSERT INTO chunks(document_id,start,end,text) VALUES(?,?,?,?) RETURNING id",
                                 (doc_id, start, end, part)).fetchone()
                db.execute("INSERT INTO chunk_vectors(rowid,embedding,tenant,collection) VALUES(?,?,?,?)",
                           (row[0], embedding, tenant, collection))
            self.audit(db, tenant, "document_ingested", doc_id)
            return {"id": doc_id, "duplicate": False, "chunks": len(parts)}

    @staticmethod
    def purge(db, tenant, doc_id=None):
        condition = "id=?" if doc_id is not None else "expires<=?"
        argument = doc_id if doc_id is not None else time.time()
        ids = [row[0] for row in db.execute(f"SELECT id FROM documents WHERE tenant=? AND {condition}", (tenant, argument))]
        for identifier in ids:
            vectors = db.execute("SELECT id FROM chunks WHERE document_id=?", (identifier,)).fetchall()
            db.executemany("DELETE FROM chunk_vectors WHERE rowid=?", vectors)
            db.execute("DELETE FROM documents WHERE id=? AND tenant=?", (identifier, tenant))
            Store.audit(db, tenant, "document_deleted" if doc_id else "document_expired", identifier)
        return bool(ids)

    def documents(self, tenant):
        with self.connect() as db:
            self.audit(db, tenant, "document_list")
            return [dict(row) for row in db.execute("""
                SELECT id,collection,title,created,expires FROM documents WHERE tenant=? AND expires>?
                ORDER BY created DESC LIMIT 100
            """, (tenant, time.time()))]

    def source(self, tenant, doc_id):
        with self.connect() as db:
            row = db.execute("SELECT id,title,collection,body FROM documents WHERE id=? AND tenant=? AND expires>?",
                             (doc_id, tenant, time.time())).fetchone()
            if row:
                self.audit(db, tenant, "source_read", doc_id)
            return dict(row) if row else None

    def delete(self, tenant, doc_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return self.purge(db, tenant, doc_id)

    def cleanup(self, tenant):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return self.purge(db, tenant)

    def search(self, tenant, collection, embedding, limit):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.purge(db, tenant)
            scope = "AND collection=?" if collection is not None else ""
            params = (embedding, tenant, limit, collection) if collection is not None else (embedding, tenant, limit)
            ranked = db.execute(f"""
                SELECT rowid,distance FROM chunk_vectors
                WHERE embedding MATCH ? AND tenant=? AND k=? {scope} ORDER BY distance
            """, params).fetchall()
            hits = []
            for item in ranked:
                row = db.execute("""
                    SELECT c.id AS chunk_id,c.start,c.end,c.text,d.id AS document_id,d.title,d.collection
                    FROM chunks c JOIN documents d ON d.id=c.document_id
                    WHERE c.id=? AND d.tenant=? AND d.expires>?
                """, (item["rowid"], tenant, time.time())).fetchone()
                if row:
                    hits.append({**dict(row), "distance": item["distance"]})
            self.audit(db, tenant, "search")
            return hits

    def memories(self, tenant):
        with self.connect() as db:
            self.audit(db, tenant, "memory_read")
            return [dict(row) for row in db.execute(
                "SELECT name,value,version FROM memory WHERE tenant=? ORDER BY name LIMIT 100", (tenant,))]

    def remember(self, tenant, name, expected, value):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT version FROM memory WHERE tenant=? AND name=?", (tenant, name)).fetchone()
            if expected != (row["version"] if row else None):
                raise Conflict("memory_version_changed")
            if value is None:
                if not row:
                    return None
                db.execute("DELETE FROM memory WHERE tenant=? AND name=?", (tenant, name))
                self.audit(db, tenant, "memory_deleted")
                return {"deleted": True}
            version = str(uuid4())
            if not row:
                db.execute("INSERT INTO memory VALUES(?,?,?,?)", (tenant, name, value, version))
            else:
                db.execute("UPDATE memory SET value=?,version=? WHERE tenant=? AND name=?",
                           (value, version, tenant, name))
            self.audit(db, tenant, "memory_written")
            return {"name": name, "value": value, "version": version}

    def audit_log(self, tenant):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT action,object_id,at FROM audit WHERE tenant=? ORDER BY id DESC LIMIT 100",
                                                   (tenant,))]
