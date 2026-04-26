"""Stub Ollama server for run-local-fast.

Returns canned responses for /api/generate and /api/embeddings so the docker
stack can be exercised end-to-end without pulling real model weights.
"""
from __future__ import annotations

import hashlib
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s stub: %(message)s")

CANNED_VISION_RESPONSE = {
    "is_first_page": True,
    "magazine": {
        "editor": "L. J. Berger",
        "issue": {"date": "1892-06-03", "volume": 5, "number": 18},
        "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place, Chicago"},
        "cost": {"issue": None, "annual": "$2.00", "semiannual": "$1.00"},
    },
    "articles": [
        {
            "title": "That's So!",
            "text": ["It is not always the man who rides the swiftest..."],
            "kind": "prose",
            "starts_on_this_page": True,
            "continues": False,
        }
    ],
}


def _deterministic_embedding(text: str, dim: int = 64) -> list[float]:
    """Hash-derived pseudo-embedding so similar strings → similar vectors (-ish)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (h * ((dim // len(h)) + 1))[:dim]
    return [(b - 128) / 128.0 for b in raw]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logging.info(fmt % args)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self._read_body()
        if self.path.startswith("/api/generate"):
            self._send({"response": json.dumps(CANNED_VISION_RESPONSE), "done": True})
        elif self.path.startswith("/api/embeddings"):
            text = body.get("prompt", "")
            self._send({"embedding": _deterministic_embedding(text)})
        else:
            self._send({"error": f"unknown path {self.path}"}, status=404)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._send({"models": [{"name": "stub:latest"}]})
        else:
            self._send({"ok": True})


def main() -> int:
    addr = ("0.0.0.0", 11434)
    logging.info("stub ollama listening on %s:%d", *addr)
    ThreadingHTTPServer(addr, Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
