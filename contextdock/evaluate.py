import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .retrieval import LocalEncoder, chunks, rank
from .store import Store

FIXTURES = {
    "Invoice schedule": "Invoice submissions close on Friday at 16:00. Late invoices move to the next cycle.",
    "Delivery window": "Delivery takes five business days. Tracking updates appear after dispatch.",
    "Access policy": "Password resets require identity verification by the help desk.",
    "Incident reporting": "Teams must record an outage in the incident log within ten minutes.",
}
QUERIES = {
    "billing submissions close": "Invoice schedule",
    "shipping tracking updates": "Delivery window",
    "credentials resets identity verification": "Access policy",
    "downtime incident log": "Incident reporting",
}


def evaluate():
    encoder = LocalEncoder()
    positions = []
    with TemporaryDirectory(prefix="contextdock-eval-") as directory:
        store = Store(Path(directory) / "eval.sqlite", encoder.identity)
        for title, body in FIXTURES.items():
            parts = list(chunks(body))
            store.ingest("evaluation", "policies", title, body, parts, encoder.encode([part[2] for part in parts]))
        for query, expected in QUERIES.items():
            candidates = store.search("evaluation", "policies", encoder.encode([query])[0], 10)
            ranked = rank(query, candidates, 4, 0.2)
            positions.append(next((i for i, row in enumerate(ranked, start=1) if row["title"] == expected), 0))
    return {"queries": len(positions), "top1_accuracy": positions.count(1) / len(positions),
            "mean_reciprocal_rank": sum(1 / rank if rank else 0 for rank in positions) / len(positions)}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
