"""A model server speaking the OpenAI routes goethe uses: health, models, chat, completions, embeddings."""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def option(args, *names):
    """The value after the first of `names` found in args, else None."""
    for name in names:
        if name in args:
            return args[args.index(name) + 1]
    return None


def serve(args):
    """Run until killed: port, key and served name come from the engine's own flags."""
    port = int(option(args, "--port"))
    key = option(args, "--api-key") or os.environ.get("GOETHE_ENGINE_API_KEY") or os.environ.get("LLAMA_API_KEY") or ""
    name = option(args, "--alias", "--served-model-name") or "model"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            return

        def send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            return not key or self.headers.get("Authorization") == f"Bearer {key}"

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, {"status": "ok"})
            if not self.allowed():
                return self.send(401, {"error": "bad key"})
            if self.path.startswith("/v1/models"):
                return self.send(200, {"object": "list", "data": [{"id": name, "object": "model"}]})
            self.send(404, {"error": "no route"})

        def do_POST(self):
            size = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(size) or b"{}")
            if not self.allowed():
                return self.send(401, {"error": "bad key"})
            if self.path.endswith("/embeddings"):
                return self.send(200, {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}], "model": name})
            if self.path.endswith("/completions") and "chat" not in self.path:
                return self.send(200, {"choices": [{"text": " ok", "index": 0}], "model": name})
            if not self.path.endswith("/chat/completions"):
                return self.send(404, {"error": "no route"})
            text = f"hello from {name}"
            if not body.get("stream"):
                return self.send(
                    200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}], "model": name}
                )
            pieces = [text[:6], text[6:], ""]
            chunks = (
                "".join(f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n" for p in pieces)
                + "data: [DONE]\n\n"
            )
            self.send(200, chunks.encode(), "text/event-stream")

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    serve(sys.argv[1:])
