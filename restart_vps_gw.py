import subprocess

cmd = "systemctl restart hermes-gateway && sleep 3 && systemctl is-active hermes-gateway && ps aux | grep hermes | grep -v grep | head -3"

result = subprocess.run(
    ["ssh", "-i", "/c/Users/dell-/.ssh/id_ed25519_vps", "-o", "StrictHostKeyChecking=no",
     "root@2.25.168.233", cmd],
    capture_output=True, text=True, timeout=20
)
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")
