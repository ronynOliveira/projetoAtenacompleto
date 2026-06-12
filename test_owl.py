import urllib.request, json

key = "sk-or-...edce"
url = "https://openrouter.ai/api/v1/chat/completions"

data = json.dumps({
    "model": "openrouter/owl-alpha",
    "messages": [{"role": "user", "content": "oi"}]
}).encode()

req = urllib.request.Request(url, data=data, headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}"
})

try:
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"Status: {resp.status}")
    body = json.loads(resp.read().decode())
    print(f"Model: {body.get('model', 'N/A')}")
    print(f"Response: {body.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')[:200]}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(e.read().decode()[:500])
except Exception as e:
    print(f"Error: {e}")
