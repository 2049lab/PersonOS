# HTTP authentication

[← HTTP service](README.md) · [Signing implementation](signing.py)

Every `/api/v1` request requires an access-key signature, including user registration and health information under that prefix. Memory operations also require `X-User-Token`, which selects the owning user's memory. An access key identifies the calling application; a user token identifies the memory owner.

The current user registry stores bearer tokens in plaintext in the database and does not expose token rotation or revocation endpoints. Account-token lifecycle management remains a deployment limitation.

The `/healthz` and `/readyz` probes, `/docs`, `/openapi.json`, and documentation redirects are outside the signed API router. The OpenAPI interface describes the routes, but its ordinary “Try it out” requests do not automatically calculate signatures.

## Server configuration

Set `PERSONOS_AKSK_MAP` before starting the service. It is a JSON object mapping access keys to signing secrets. Use your own values; the following are placeholders:

```bash
export PERSONOS_AKSK_MAP='{"local-client":"replace-with-your-signing-secret"}'
export PERSONOS_SIGN_WINDOW_S=300
```

A missing or empty key map rejects signed API requests. A nonempty map is cached in process; restart the service when replacing existing keys. The timestamp window defaults to 300 seconds and is read at import time. Client and server clocks must stay within that window.

There is **no environment-variable bypass** for signature verification. Tests can explicitly override the FastAPI `verify_signature` dependency; application deployments cannot disable it with a configuration switch.

## Sign a request

The source repository includes `server.signing.sign_request`, which implements the same canonicalization used by the server. Prefer that helper to an independent implementation when writing a Python client.

| Header | Value |
|---|---|
| `X-Access-Key` | An access key present in the server map |
| `X-Timestamp` | Integer Unix time in seconds |
| `X-Nonce` | A fresh random value for this request |
| `X-Signed-Headers` | Comma-separated names of the headers covered by the signature |
| `X-Signature` | Lowercase hexadecimal HMAC-SHA256 signature |
| `X-User-Token` | User token for memory operations; omit during registration |

When present and nonempty, `content-type` and `x-user-token` must be signed. The helper lowercases header names, signs those two headers when supplied, and generates a timestamp plus a random 16-byte nonce. Keep the method, path, query values, headers and body unchanged after signing.

The canonical string has these components joined with newline characters:

```text
UPPERCASE_METHOD
request_path
sorted_query_pairs
signed_header_lines
sha256_of_exact_body_bytes
timestamp
nonce
```

The implementation builds the components as follows:

1. Use the request path, including `/api/v1`, without the scheme, host or query string.
2. Sort the decoded `(name, value)` query pairs lexicographically, including repeated parameters, then join each `name=value` pair with `&`. The current canonicalizer does not percent-encode these decoded values.
3. Sort the lowercase signed header names and join `name:trimmed-value` entries with newlines. Use an empty string when there are no signed headers.
4. Hash the exact bytes sent as the body with SHA-256 and use the lowercase hexadecimal digest. A bodyless request hashes empty bytes.
5. Compute HMAC-SHA256 over the UTF-8 canonical string with the UTF-8 signing secret; transmit its lowercase hexadecimal digest.

For a retry, sign again with a fresh nonce. Nonces are tracked per access key for the configured timestamp window. Redis coordinates this across workers when available; the fallback is process-local, including when a Redis operation fails. This fallback does not provide cross-worker replay protection.

## Python example

Run from the repository root after installing the library and server dependencies. Start the server with the key map above, then set the same key and secret in the client environment:

```bash
export PERSONOS_CLIENT_ACCESS_KEY="local-client"
export PERSONOS_CLIENT_SECRET_KEY="replace-with-your-signing-secret"
```

This example registers a new example user, then reads an empty memory page. It makes no model request.

```python
import json
import os
import uuid

import httpx

from server.signing import sign_request

access_key = os.environ["PERSONOS_CLIENT_ACCESS_KEY"]
secret_key = os.environ["PERSONOS_CLIENT_SECRET_KEY"]


def request(client, method, path, *, payload=None, token=None, params=None):
    body = b"" if payload is None else json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    headers = {"content-type": "application/json"} if payload is not None else {}
    if token:
        headers["x-user-token"] = token
    query_items = list(params or [])
    headers.update(sign_request(
        access_key, secret_key, method, path,
        query_items=query_items, headers=headers, body=body,
    ))
    response = client.request(
        method, path, params=query_items, content=body, headers=headers,
    )
    response.raise_for_status()
    return response.json()["data"]


with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30) as client:
    user = request(
        client, "POST", "/api/v1/users/register",
        payload={"user_id": "example-" + uuid.uuid4().hex},
    )
    token = user["token"]  # Retain it securely; registration returns it once.
    page = request(
        client, "GET", "/api/v1/memories",
        token=token, params=[("limit", "20")],
    )
    print(page["items"])
```

The example leaves its registered user in the database. For a real application, store and reuse the user's token instead of registering a new user on every request.

## Errors

Signature failures return HTTP `401` with `msg: "authentication failed"`. The service does not expose whether the key, nonce, timestamp or signature caused the failure. A missing or invalid user token also returns `401` but is checked separately after request signing.

Use TLS for network deployments. Signing authenticates request contents; it does not encrypt the body or user token.
