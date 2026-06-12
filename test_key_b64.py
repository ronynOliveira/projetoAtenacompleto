import base64, urllib.request, json

# Decode the key from base64
key_b64 = "***"
key = base64.b64decode(key_b64).decode()

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {key}"}
)
try:
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read().decode())
    d = data.get("data", {})
    print(f"OK: limit_remaining={d.get('limit_remaining','N/A')}, usage={d.get('usage','N/A')}, free_tier={d.get('is_free_tier','N/A')}")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:200]}")
except Exception as e:
    print(f"Error: {e}")
