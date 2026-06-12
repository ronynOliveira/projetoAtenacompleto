import subprocess, base64

key = "sk-or-...edce"
key_b64 = base64.b64encode(key.encode()).decode()

# Write a script on VPS that decodes the key
cmd = f"echo '{key_b64}' | base64 -d > /tmp/or_key.txt && cat /tmp/or_key.txt"

result = subprocess.run(
    ["ssh", "-i", "/c/Users/dell-/.ssh/id_ed25519_vps", "-o", "StrictHostKeyChecking=no",
     "root@2.25.168.233", cmd],
    capture_output=True, text=True, timeout=15
)
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")
print(f"exit: {result.returncode}")
