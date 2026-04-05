#!/usr/bin/env python3
"""
dev-server.py — Tiny save server for the dev-editor.
Accepts POST /save with JSON {path, html}, backs up the original file,
then replaces <article>...</article> content and writes it back.

Usage:  python3 site/scripts/dev-server.py
Runs on port 8081 with CORS headers for localhost:8080.
"""

import json
import re
import shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parent.parent  # site/


class SaveHandler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/save":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        page_path: str = body["path"]   # e.g. "/site/lectures/lecture09.html"
        # Strip leading /site/ prefix if serving from repo root
        page_path = re.sub(r"^/site/", "", page_path.lstrip("/"))
        new_html: str = body["html"]

        # Resolve to a file under SITE_ROOT
        file_path = (SITE_ROOT / page_path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(SITE_ROOT)):
            self.send_response(403)
            self.end_headers()
            return

        if not file_path.exists():
            self._respond(404, f"File not found: {file_path}")
            return

        # Backup original
        backup = file_path.with_suffix(".html.bak")
        shutil.copy2(file_path, backup)

        # Replace article content
        original = file_path.read_text(encoding="utf-8")
        pattern = re.compile(
            r"(<article[^>]*>)(.*?)(</article>)", re.DOTALL
        )
        if not pattern.search(original):
            self._respond(400, "No <article> tag found in file")
            return

        updated = pattern.sub(rf"\1\n{new_html}\n\3", original, count=1)
        file_path.write_text(updated, encoding="utf-8")

        self._respond(200, "OK")
        print(f"  Saved {file_path.relative_to(SITE_ROOT)} ({len(new_html)} chars)")

    def _respond(self, code: int, msg: str) -> None:
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(msg.encode())

    def log_message(self, fmt, *args) -> None:  # type: ignore[override]
        print(f"[dev-server] {args[0]}")


if __name__ == "__main__":
    server = HTTPServer(("localhost", 8081), SaveHandler)
    print(f"Dev save server running on http://localhost:8081")
    print(f"Site root: {SITE_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
