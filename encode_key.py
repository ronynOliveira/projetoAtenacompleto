import base64
key = "sk-or-...edce"
encoded = base64.b64encode(key.encode()).decode()
print(encoded)
