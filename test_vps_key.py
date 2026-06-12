import urllib.request, json
import os

key = os.environ.get("OPENROUTER_API_KEY", "")
if not key:
    # Try to read from .env
    env_path = "/root/.hermes/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.strip().split("=", 1)[1]
                    break

if not key:
    print("ERROR: No key found")
    exit(1)

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
