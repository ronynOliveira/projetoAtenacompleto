# SESSÃO 2026-06-14 — EPR Bridge: Correção e Ativação
**Data:** 14/06/2026
**Status:** EPR Bridge FUNCIONAL e operacional

---

## PROBLEMAS RESOLVIDOS

### 1. HMAC Secret Mismatch
- **Problema:** Servidor usava secret default (`default-insecure-koldi-secret-key-change-me`) porque `EPR_SECRET` não estava setado no systemd
- **Solução:** Adicionado `Environment=EPR_SECRET=epr-bridge-2026-secret-key-koldi-secret-key-koldi-fusion` no systemd service
- **Resultado:** HMAC signature validada corretamente

### 2. Sync Paths Duplicados
- **Problema:** `epr_server.json` tinha paths absolutos (`/opt/hermes/.hermes/lib`) que eram concatenados com `self.root` (`/opt/hermes`), resultando em `/opt/hermes/opt/hermes/.hermes/lib`
- **Solução:** Alterados para paths relativos (`.hermes/lib`, `.hermes/wiki`, etc.)
- **Resultado:** Watchdog monitora paths corretos

### 3. Arquivos de Identidade
- **Problema:** SOUL.md e USER.md eram diretórios vazios na VPS (bug do Unison)
- **Solução:** Removidos diretórios, copiados arquivos reais via SCP
- **Resultado:** SOUL.md (522 linhas, v4.3) e USER.md (58 linhas) presentes

### 4. Wiki Vazia na VPS
- **Problema:** Wiki não sincronizava porque `path = wiki` não estava no perfil Unison
- **Solução:** Adicionado `path = wiki` ao `koldi.prf` local e copiado para VPS
- **Resultado:** 124 arquivos wiki sincronizados

### 5. MEMORY.md era diretório
- **Problema:** Unison criou diretório ao invés de arquivo
- **Solução:** Removido diretório, copiado arquivo real

---

## ESTADO ATUAL DO EPR BRIDGE

### VPS (2.25.168.233)
- ✅ Servidor WebSocket TLS rodando (PID 14644, porta 8443)
- ✅ systemd service: `epr-bridge` (enabled + running)
- ✅ Watchdog monitorando 6 paths corretos
- ✅ 956 arquivos rastreados
- ✅ EPR_SECRET configurado no systemd
- ✅ SSL cert fingerprint: fc742df09668dfbe112ca7d5e45e3aa3dbabe1f5c1a9c80995bec73d9766edb2

### Local (Windows)
- ✅ Cliente EPR configurado e funcional
- ✅ Conexão WebSocket com HMAC validada
- ✅ Cert pinning funcionando

---

## ARQUIVOS MODIFICADOS
1. `/etc/systemd/system/epr-bridge.service` — adicionado EPR_SECRET
2. `/opt/hermes/.hermes/epr_server.json` — paths relativos
3. `C:\Users\dell-\.unison\koldi.prf` — adicionado path wiki
4. `~/.unison/koldi.prf` na VPS — copiado do local

---

## PRÓXIMOS PASSOS
- [ ] Testar sincronização bidirecional (push de arquivo local → VPS)
- [ ] Testar detecção de mudanças em tempo real (modificar arquivo → sync automático)
- [ ] Testar resolução de conflitos
- [ ] Configurar cliente EPR para rodar automaticamente (systemd/cron)
- [ ] Integrar com KCPA (padrões de comunicação via ponte)
- [ ] Documentar protocolo EPR no wiki

---

*Sessão salva em 14/06/2026*
