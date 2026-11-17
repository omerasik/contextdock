import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .retrieval import LocalEncoder, chunks, excerpts, rank
from .store import Store

FIXTURES = {
    "Invoice schedule": "Invoice submissions close on Friday at 16:00. Late invoices move to the next cycle.",
    "Delivery window": "Delivery takes five business days. Tracking updates appear after dispatch.",
    "Access policy": "Password resets require identity verification by the help desk.",
    "Incident reporting": "Teams must record an outage in the incident log within ten minutes.",
    "Refund policy": "Refund requests require proof of purchase and must arrive within fourteen calendar days. "
    "The support team checks the order number and refund reason. Refunds use the original payment method; "
    "staff must never ask customers to share a card password.",
    "Procurement approval": "Procurement reviews require an approved vendor and a written quote. "
    "A quote above the spending limit needs a manager decision. The internal catalog records vendor approval; "
    "a recommendation is not a purchase order or permission to send payment.",
    "Backup restoration": "Encrypted backups are kept for thirty-five days. The backup owner records the snapshot "
    "identifier and verifies the archive checksum before a restoration begins. A restoration is tested in an isolated "
    "environment before any production data is changed. Staff use a temporary test database and compare document counts "
    "with the snapshot manifest. Each discrepancy receives a written investigation note. The archive encryption key "
    "is managed outside the backup storage account. Access requires the on-call operator and a second reviewer. "
    "A monthly restore drill checks that a recent snapshot can be restored. The drill records the chosen snapshot, "
    "the operator, validation results, and any missing records. A failed drill creates a follow-up task. "
    "The monthly drill result is reviewed at the next operations meeting. Restore duration is measured in the drill "
    "and must not be replaced with an invented service guarantee.",
    "Document retention": "Retained source documents expire after ninety days. Expiration hides the content from "
    "search immediately. Scheduled cleanup removes the stored source and vector index entries together. "
    "A tenant may delete a document earlier. Database backups have a separate retention policy.",
    "Expense receipts": "Expense claims require an itemized receipt and a short business purpose. "
    "Submit the receipt within ten business days of the expense. The finance reviewer checks the date, currency "
    "and total before reimbursement. A card statement alone is not an itemized receipt.",
    "Onboarding checklist": "New staff receive an account owner and an onboarding buddy on their first day. "
    "The buddy explains the shared calendar and support channels. Access is granted by the account owner "
    "after the role checklist is reviewed. Temporary access must have an expiry date. The newcomer signs "
    "into a test workspace before using business records. Staff do not share passwords during onboarding. "
    "The buddy checks that the newcomer can find the written help instructions. A missing permission is "
    "reported to the account owner rather than solved with a colleague's credentials. The first-week check "
    "also covers document retention, support escalation and the location of approved templates. The buddy "
    "records open questions in the onboarding note. A welcome meeting takes place at the ground-floor café. "
    "The meeting location is included in the calendar invitation. The newcomer may request an accessible "
    "alternative location from the buddy.",
    "Remote work": "Remote work schedules must show at least three overlapping collaboration hours. "
    "The team calendar records the employee's timezone. Changes to the agreed schedule are discussed "
    "with the team lead before the new schedule starts. Availability does not require sharing a home address.",
    "Security reporting": "Report suspicious links and phishing messages to the security team. "
    "Include the sender, arrival time and a screenshot when available. Do not open an attachment to gather evidence. "
    "The security team triages the report and tells the employee which message can be removed.",
}
QUERIES = {
    "billing submissions close": "Invoice schedule",
    "shipping tracking updates": "Delivery window",
    "credentials resets identity verification": "Access policy",
    "downtime incident log": "Incident reporting",
    "late invoice next cycle": "Invoice schedule",
    "delivery five business days": "Delivery window",
    "password reset help desk": "Access policy",
    "record outage ten minutes": "Incident reporting",
    "refund proof purchase": "Refund policy",
    "refund fourteen calendar days": "Refund policy",
    "procurement approved vendor written quote": "Procurement approval",
    "quote spending limit manager": "Procurement approval",
    "encrypted backups thirty five days": "Backup restoration",
    "monthly restore drill validation": "Backup restoration",
    "source documents expire ninety days": "Document retention",
    "cleanup source vector index": "Document retention",
    "expense itemized receipt business purpose": "Expense receipts",
    "submit expense ten business days": "Expense receipts",
    "new staff onboarding buddy first day": "Onboarding checklist",
    "welcome meeting ground floor café": "Onboarding checklist",
    "remote work overlapping collaboration hours": "Remote work",
    "calendar employee timezone schedule": "Remote work",
    "suspicious links phishing messages": "Security reporting",
    "report sender screenshot security team": "Security reporting",
}
NEGATIVES = ["volcano basalt magma", "violin concerto orchestra", "galaxy nebula telescope", "saffron recipe simmer"]


def evaluate():
    encoder = LocalEncoder()
    positions = []
    results = []
    no_answers = 0
    chunk_count = 0
    with TemporaryDirectory(prefix="contextdock-eval-") as directory:
        store = Store(Path(directory) / "eval.sqlite", encoder.identity)
        for title, body in FIXTURES.items():
            parts = list(chunks(body))
            chunk_count += len(parts)
            store.ingest("evaluation", "policies", title, body, parts, encoder.encode([part[2] for part in parts]))
        for query, expected in QUERIES.items():
            candidates = store.search("evaluation", "policies", encoder.encode([query])[0], 30)
            ranked = rank(query, candidates, 10, 0.2)
            titles = list(dict.fromkeys(row["title"] for row in ranked))
            position = titles.index(expected) + 1 if expected in titles else 0
            positions.append(position)
            results.append({"query": query, "expected": expected, "rank": position,
                            "top_source": titles[0] if titles else None})
        for query in NEGATIVES:
            candidates = store.search("evaluation", "policies", encoder.encode([query])[0], 30)
            no_answers += not excerpts(query, rank(query, candidates, 5, 0.2))
    return {"dataset": "authored-policies-v2", "documents": len(FIXTURES), "chunks": chunk_count,
            "queries": len(positions), "negative_queries": len(NEGATIVES),
            "top1_accuracy": positions.count(1) / len(positions),
            "recall_at_3": sum(0 < position <= 3 for position in positions) / len(positions),
            "mean_reciprocal_rank": sum(1 / position if position else 0 for position in positions) / len(positions),
            "no_answer_accuracy": no_answers / len(NEGATIVES), "results": results}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
