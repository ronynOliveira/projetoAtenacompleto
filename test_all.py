import urllib.request, json

key = "sk-or-...edce"

# Test 1: /key endpoint
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {key}"}
)
try:
    resp = urllib.request.urlopen(req, timeout=10)
    print(f"/key: {resp.status}")
    d = json.loads(resp.read().decode())
    print(json.dumps(d.get("data", {}), indent=2)[:300])
except urllib.error.HTTPError as e:
    print(f"/key: HTTP {e.code}: {e.read().decode()[:200]}")

# Test 2: /models endpoint
req2 = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {key}"}
)
try:
    resp2 = urllib.request.urlopen(req2, timeout=10)
    print(f"/models: {resp2.status}")
except urllib.error.HTTPError as e:
    print(f"/models: HTTP {e.code}")

# Test 3: chat with different key format
data = json.dumps({
    "model": "openrouter/owl-alpha",
    "messages": [{"role": "user", "content": "oi"}]
}).encode()

req3 = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=data,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://koldi.local",
        "X-Title": "Koldi"
    }
)
try:
    resp3 = urllib.request.urlopen(req3, timeout=30)
    print(f"chat: {resp3.status}")
except urllib.error.HTTPError as e:
    print(f"chat: HTTP {e.code}: {e.read().decode()[:200]}")
