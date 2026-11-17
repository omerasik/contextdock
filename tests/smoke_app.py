"""Verify tenant-scoped retrieval and memory against a local demo server."""

import argparse
import json
import time
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8030")
    args = parser.parse_args()
    origin = args.url.rstrip("/")
    if urlparse(origin).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Use a loopback demo server.")

    def call(user, path, body=None, method=None):
        request = Request(origin + "/api/" + path, method=method,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={"Authorization": "Bearer contextdock-demo-" + user, "Content-Type": "application/json"})
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    for attempt in range(30):
        try:
            call("alice", "health")
            break
        except (URLError, ConnectionError, TimeoutError, HTTPException):
            if attempt == 29:
                raise
            time.sleep(1)
    records = {}
    for user, day in [("alice", "Friday"), ("bob", "Monday")]:
        records[user] = call(user, "documents", {"collection": "smoke", "title": user + " policy",
                                               "text": "Invoice submissions close on " + day + "."})["id"]
        result = call(user, "search", {"query": "billing submissions close", "collection": "smoke"})
        assert result["answer_mode"] == "extractive"
        assert {source["document_id"] for source in result["sources"]} == {records[user]}
        assert day in result["answer"][0]["quote"]
    for user, other in [("alice", "bob"), ("bob", "alice")]:
        try:
            call(user, "documents/" + records[other])
            raise AssertionError("Cross-tenant source was exposed")
        except HTTPError as error:
            assert error.code == 404
    initial = call("alice", "memory/smoke", {"value": "Dutch"}, "PUT")
    assert not any(item["name"] == "smoke" for item in call("bob", "memory"))
    updated = call("alice", "memory/smoke", {"value": "English", "expected": initial["version"]}, "PUT")
    assert updated["version"] != initial["version"]
    assert call("alice", "memory/smoke", {"value": None, "expected": updated["version"]}, "PUT")["deleted"]
    for user in records:
        assert call(user, "documents/" + records[user], method="DELETE")["deleted"]
        assert call(user, "search", {"query": "billing submissions close", "collection": "smoke"})["sources"] == []
    print("PASS: actual vector index, two-tenant sources/citations, private memory updates and deletion")


if __name__ == "__main__":
    main()
