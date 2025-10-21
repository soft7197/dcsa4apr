"""Start CodeLlama-34B-Instruct as a local API server on 2x A6000 GPUs.

Architecture
------------
  client (port 8000)  →  proxy [this process]  →  vLLM (port 8001)

The built-in proxy fixes CodeLlama-specific failure modes:
  • JSON schema enforcement  – forces the model to emit the correct keys/types
  • Response coercion        – post-processes remaining type mismatches
  • Timeout retry            – retries slow vLLM calls automatically

Run once, stays loaded in GPU memory. Send queries from anywhere via query.py.
"""

import asyncio
import json
import os
import subprocess
import sys
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
try:
    from json_repair import repair_json as _repair_json
except ImportError:
    _repair_json = None

# ---------------------------------------------------------------------------
# Model / server config
# ---------------------------------------------------------------------------
MODEL     = "codellama/CodeLlama-34b-Instruct-hf"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "CodeLlama-34b-Instruct-hf")

VLLM_PORT  = 8001          # internal port — vLLM listens here
PROXY_PORT = 8000          # client-facing port — unchanged for callers
HOST       = "0.0.0.0"

# ---------------------------------------------------------------------------
# Proxy config
# ---------------------------------------------------------------------------
VLLM_BASE_URL  = f"http://localhost:{VLLM_PORT}"
MAX_RETRIES    = 3
REQUEST_TIMEOUT = 1800.0   # 30 min — enough for n=10 on a 34B model

# ---------------------------------------------------------------------------
# JSON schemas that match the two prompt formats in generator_agents.py
# ---------------------------------------------------------------------------
_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis":   {"type": "string"},
        "changes":      {"type": "string"},
        "fixed_method": {"type": "string", "minLength": 1},
    },
    "required": ["hypothesis", "changes", "fixed_method"],
    # No additionalProperties constraint — keeps FSA small and generation fast
}

_MULTI_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "changes":    {"type": "string"},
        "methods": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "method_name":  {"type": "string"},
                    "fixed_method": {"type": "string", "minLength": 1},
                },
                "required": ["fixed_method"],
            },
        },
    },
    "required": ["hypothesis", "methods"],
    # No additionalProperties constraint — keeps FSA small and generation fast
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_multi(messages: list) -> bool:
    return any(
        "Multi-Function Bug Fixing Task" in m.get("content", "")
        or "Multi-Method Component" in m.get("content", "")
        for m in messages
    )


def _to_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    return str(val) if not isinstance(val, str) else val


def _extract_by_braces(content: str, field: str) -> str:
    """Extract a Java method body from raw LLM content by tracking brace depth,
    bypassing JSON string parsing entirely.  Handles unescaped double quotes that
    would truncate the value under normal json.loads().
    """
    marker = f'"{field}"'
    idx = content.find(marker)
    if idx == -1:
        return ""
    idx += len(marker)
    # skip whitespace and ':'
    while idx < len(content) and content[idx] in ' \t\n\r:':
        idx += 1
    if idx >= len(content) or content[idx] != '"':
        return ""
    idx += 1  # skip opening quote

    result = []
    depth = 0
    while idx < len(content):
        c = content[idx]
        if c == '{':
            depth += 1
        elif c == '}':
            if depth == 0:
                break
            depth -= 1
            result.append(c)
            if depth == 0:
                break
            idx += 1
            continue
        result.append(c)
        idx += 1

    return ''.join(result).strip()


def _coerce(content: str, multi: bool) -> str:
    """Fix type mismatches in a single choice's content string."""
    try:
        obj = json.loads(content)
    except Exception:
        # Try json_repair (handles single-quoted JSON, minor truncation, etc.)
        if _repair_json is not None:
            try:
                obj = json.loads(_repair_json(content))
            except Exception:
                obj = None
        else:
            obj = None

    # Unwrap array: model sometimes returns [{...}] instead of {...}
    if isinstance(obj, list):
        obj = obj[0] if obj and isinstance(obj[0], dict) else {}

    if not isinstance(obj, dict):
        obj = {}

    if multi:
        methods = obj.get("methods", [])
        if isinstance(methods, dict):
            methods = list(methods.values())
        elif not isinstance(methods, list):
            methods = []
        # Normalise each item: must be a dict with fixed_method string
        normalised = []
        for m in methods:
            if isinstance(m, dict):
                m["fixed_method"] = _to_str(m.get("fixed_method"))
                normalised.append(m)
            elif isinstance(m, list):
                normalised.append({"fixed_method": "\n".join(_to_str(x) for x in m)})
            elif isinstance(m, str):
                normalised.append({"fixed_method": m})
        obj["methods"] = normalised
        for k in ("hypothesis", "changes"):
            obj[k] = _to_str(obj.get(k))
    else:
        for k in ("hypothesis", "changes", "fixed_method"):
            obj[k] = _to_str(obj.get(k))

        # If fixed_method looks truncated (no closing brace), try brace-based extraction
        # directly from the raw content — recovers from unescaped double quotes
        fm = obj.get("fixed_method", "")
        if not fm or ('{' in fm and not fm.rstrip().endswith('}')):
            recovered = _extract_by_braces(content, "fixed_method")
            if recovered and len(recovered) > len(fm):
                obj["fixed_method"] = recovered

    return json.dumps(obj)


# ---------------------------------------------------------------------------
# FastAPI proxy app
# ---------------------------------------------------------------------------
app = FastAPI()


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    body_bytes = await request.body()
    multi = False

    # Intercept chat/completions to inject JSON schema and speed limits
    if path == "chat/completions" and body_bytes:
        try:
            body    = json.loads(body_bytes)
            msgs    = body.get("messages", [])
            multi   = _is_multi(msgs)
            schema  = _MULTI_SCHEMA if multi else _SINGLE_SCHEMA

            if body.get("response_format", {}).get("type") == "json_object":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "patch_response", "schema": schema, "strict": True},
                }

            if body.get("n", 1) > 10:
                body["n"] = 10

            # Cap max_tokens to avoid runaway completions
            # single: ~800 tokens covers any realistic method body
            # multi:  ~1500 tokens for 3 method bodies with some margin
            if "max_tokens" not in body:
                body["max_tokens"] = 1500 if multi else 800

            body_bytes = json.dumps(body).encode()
        except Exception:
            pass  # forward as-is if anything goes wrong

    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length")}

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.request(
                    method=request.method,
                    url=f"{VLLM_BASE_URL}/v1/{path}",
                    content=body_bytes,
                    headers=fwd_headers,
                )

            if path == "chat/completions" and resp.status_code == 200:
                result = resp.json()
                for choice in result.get("choices", []):
                    raw = choice.get("message", {}).get("content", "")
                    choice["message"]["content"] = _coerce(raw, multi)
                return JSONResponse(content=result, status_code=200)

            try:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            except Exception:
                return Response(content=resp.content, status_code=resp.status_code,
                                media_type=resp.headers.get("content-type"))

        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "proxy_error"}}, status_code=500
            )

    return JSONResponse(
        {"error": {"message": f"Timed out after {MAX_RETRIES} attempts: {last_exc}",
                   "type": "timeout"}},
        status_code=504,
    )


# ---------------------------------------------------------------------------
# Startup: download → vLLM (background) → wait → proxy (foreground)
# ---------------------------------------------------------------------------
def _wait_for_vllm(timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{VLLM_BASE_URL}/health", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


if __name__ == "__main__":
    # Step 1: Download model locally (skip if already exists)
    if os.path.isdir(LOCAL_DIR) and any(f.endswith(".safetensors") for f in os.listdir(LOCAL_DIR)):
        print(f"Model already exists at {LOCAL_DIR}, skipping download.\n")
    else:
        print(f"Downloading {MODEL} to {LOCAL_DIR} ...")
        snapshot_download(repo_id=MODEL, local_dir=LOCAL_DIR)
        print(f"Model ready at {LOCAL_DIR}\n")

    # Step 2: Start vLLM on the internal port (background process)
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  LOCAL_DIR,
        "--served-model-name",      MODEL,
        "--tensor-parallel-size",   "2",
        "--dtype",                  "bfloat16",
        "--gpu-memory-utilization", "0.95",
        "--max-model-len",          "16384",
        "--host",                   HOST,
        "--port",                   str(VLLM_PORT),
        "--enable-prefix-caching",   # cache repeated system-prompt KV across requests
        "--enable-chunked-prefill",  # better GPU utilisation for mixed prompt lengths
        # xgrammar is the default structured-outputs backend in vLLM ≥ 0.15
    ]
    print(f"Starting vLLM on internal port {VLLM_PORT} ...")
    vllm_proc = subprocess.Popen(vllm_cmd)

    # Step 3: Wait until vLLM is ready
    print("Waiting for vLLM to be ready ...")
    if not _wait_for_vllm(timeout=300):
        print("ERROR: vLLM did not start within 5 minutes.")
        vllm_proc.terminate()
        sys.exit(1)
    print("vLLM is ready.\n")

    # Step 4: Start the proxy on the client-facing port (foreground)
    print(f"Starting proxy on http://{HOST}:{PROXY_PORT} ...")
    print("Model loads once. Send queries from another terminal using query.py\n")
    try:
        uvicorn.run(app, host=HOST, port=PROXY_PORT, log_level="info")
    finally:
        vllm_proc.terminate()
