import urllib.request, json, os
from pathlib import Path

# Read .env manually
env_path = Path("/root/.hermes/.env")
key = None
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.strip().startswith("OPENROUTER_API_KEY="):
            k = line.strip().split("=", 1)[1]
            if k and not k.startswith("***"):
                key = k
                break

if not key:
    # Try environment
    key = os.environ.get("OPENROUTER_API_KEY", "")

if not key:
    print("ERROR: No key found")
    exit(1)

# Mask for logging
print(f"Key: {key[:10]}...{key[-4:]} (len={len(key)})")

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
