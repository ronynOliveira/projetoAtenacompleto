import subprocess
result = subprocess.run(
    ["ssh", "-i", "/c/Users/dell-/.ssh/id_ed25519_vps", "-o", "StrictHostKeyChecking=no",
     "root@2.25.168.233",
     "grep -c OPENROUTER_API_KEY /root/.hermes/.env"],
    capture_output=True, text=True, timeout=10
)
print(f"count: {result.stdout.strip()}")
print(f"err: {result.stderr.strip()}")
