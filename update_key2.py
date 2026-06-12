import sys
sys.path.insert(0, r'C:\Users\dell-\AppData\Local\hermes\scripts')
from cofre import _get_cipher, VAULT_FILE, load_state, hash_password
import json

senha = "EW8&mRwss%SH3E9ZFpj9e@#l"
chave = "OPENROUTER_API_KEY"
valor = "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"

state = load_state()
if hash_password(senha) != state.get("password_hash", ""):
    print("Senha incorreta")
    sys.exit(1)

cipher = _get_cipher(senha)
encrypted_data = VAULT_FILE.read_bytes()
decrypted = cipher.decrypt(encrypted_data)
data = json.loads(decrypted.decode())

data[chave] = valor

new_encrypted = cipher.encrypt(json.dumps(data).encode())
VAULT_FILE.write_bytes(new_encrypted)
print("OK - chave atualizada")