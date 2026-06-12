import subprocess

# Append the key to .env using the temp file
cmd = "echo \"OPENROUTER_API_KEY=$(cat /tmp/or_key.txt)\" >> /root/.hermes/.env && echo \"OPENAI_API_KEY=$(cat /tmp/or_key.txt)\" >> /root/.hermes/.env && grep OPENROUTER /root/.hermes/.env | tail -2"

result = subprocess.run(
    ["ssh", "-i", "/c/Users/dell-/.ssh/id_ed25519_vps", "-o", "StrictHostKeyChecking=no",
     "root@2.25.168.233", cmd],
    capture_output=True, text=True, timeout=15
)
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")
