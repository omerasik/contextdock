import hashlib
import json
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import httpx2

DIMENSIONS = 128
STOP = frozenset("the an a in from of is are to my how what where when and or for after before it can i".split())
ALIASES = {
    "billing": "invoice", "payment": "invoice", "invoices": "invoice",
    "shipping": "delivery", "shipped": "delivery", "shipment": "delivery",
    "outages": "outage", "downtime": "outage", "interruption": "outage",
    "passwords": "password", "credentials": "password",
}


def terms(text: str) -> list[str]:
    return [ALIASES.get(word, word) for word in re.findall(r"[^\W_]+", text.casefold()) if word not in STOP]


def chunks(text: str):
    start = 0
    while start < len(text):
        stop = min(start + 720, len(text))
        if stop < len(text):
            boundary = text.rfind(" ", start + 500, stop)
            if boundary >= start + 500:
                stop = boundary
        if part := text[start:stop].strip():
            offset = text.find(part, start, stop)
            yield offset, offset + len(part), part
        if stop >= len(text):
            break
        start = stop - 90


def vector(values: list[float]) -> bytes:
    if len(values) != DIMENSIONS or not all(type(value) in (int, float) and math.isfinite(value) for value in values):
        raise ValueError("invalid_embedding")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or not norm:
        raise ValueError("empty_embedding")
    return struct.pack(f"<{DIMENSIONS}f", *(value / norm for value in values))


class Encoder(Protocol):
    identity: str

    def encode(self, texts: list[str]) -> list[bytes]: ...


class LocalEncoder:
    identity = "local-hash-v1"

    def encode(self, texts: list[str]) -> list[bytes]:
        encoded = []
        for text in texts:
            weights = [0.0] * DIMENSIONS
            for word, count in Counter(terms(text)).items():
                hashed = hashlib.blake2s(word.encode(), digest_size=4).digest()
                index = int.from_bytes(hashed, "little") % DIMENSIONS
                weights[index] += (1 + math.log(count)) * (1 if hashed[2] & 1 else -1)
            if not any(weights):
                raise ValueError("no_searchable_terms")
            encoded.append(vector(weights))
        return encoded


@dataclass(frozen=True)
class RemoteEncoder:
    endpoint: str
    model: str
    token: str

    @property
    def identity(self):
        return f"{self.endpoint}#{self.model}"

    def encode(self, texts: list[str]) -> list[bytes]:
        try:
            with httpx2.stream("POST", self.endpoint, headers={"Authorization": f"Bearer {self.token}"},
                               json={"model": self.model, "input": texts, "dimensions": DIMENSIONS},
                               timeout=8, follow_redirects=False) as response:
                response.raise_for_status()
                body = bytearray()
                for item in response.iter_bytes(chunk_size=8192):
                    body.extend(item)
                    if len(body) > 512_000:
                        raise ValueError("embedding_response_size")
            rows = sorted(json.loads(body)["data"], key=lambda row: row["index"])
            if any(type(row["index"]) is not int for row in rows) or [row["index"] for row in rows] != list(range(len(texts))):
                raise ValueError("embedding_index_mismatch")
            return [vector(row["embedding"]) for row in rows]
        except (httpx2.HTTPError, ValueError, TypeError, KeyError, OverflowError) as error:
            raise RuntimeError("embedding_provider_unavailable") from error


def rank(question: str, candidates: list[dict], limit: int, threshold: float):
    query = set(terms(question))
    for row in candidates:
        confidence = max(0.0, 1 - row.pop("distance"))
        overlap = len(query.intersection(terms(row["text"]))) / max(1, len(query))
        row["score"] = round(0.75 * confidence + 0.25 * overlap, 4)
    return sorted((row for row in candidates if row["score"] >= threshold),
                  key=lambda row: (-row["score"], row["chunk_id"]))[:limit]


def excerpts(query: str, sources: list[dict]):
    normalized = set(terms(query))
    answer = []
    for row in sources[:3]:
        for match in re.finditer(r"[^.!?\n]+[.!?]?", row["text"]):
            sentence = match.group().strip()
            if sentence and normalized.intersection(terms(sentence)):
                start = row["start"] + match.start() + len(match.group()) - len(match.group().lstrip())
                answer.append({"quote": sentence, "document_id": row["document_id"], "chunk_id": row["chunk_id"],
                               "start": start, "end": start + len(sentence)})
                break
    return answer
