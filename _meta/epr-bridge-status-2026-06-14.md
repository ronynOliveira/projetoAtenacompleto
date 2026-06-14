# EPR Bridge — Status 2026-06-14

## Estado Operacional

### Cliente Local (Windows 10)
- **PID:** 11400 (pythonw.exe)
- **Modo:** client
- **Conexao:** WebSocket SSL para wss://2.25.168.233:8443
- **Banco:** G:\Meu Drive\Koldi\.hermes\.epr_state_local.db
- **Arquivos rastreados:** 1628
- **Synced:** ~1010
- **Pending:** ~618 (skills/templates nao sincronizados ainda)
- **Conflict:** 0
- **__pycache__ removidos do banco:** 173+71+58+41 = 343 entradas limpas

### Servidor VPS (Debian)
- **PID:** 24460
- **Modo:** server
- **Banco:** /opt/hermes/.hermes/.epr_state.db
- **Arquivos rastreados:** 1519
- **Pending (lado servidor):** 148

## Problemas Resolvidos
1. **Erro de ortografia:** "Nao existe" -> "Nao existe" (linha 371) - corrigido
2. **__pycache__ no banco:** 343 entradas removidas (ignore_pattern funciona para novas, mas antigas precisam de limpeza manual)
3. **Cliente EPR nao subia:** CREATE_NO_WINDOW + DETACHED_PROCESS funciona, mas precisa de cwd com caminho Windows nativo
4. **Status 'synced' nao atualizado:** Corrigido na linha 569-583 (update_file_state apos envio bem-sucedido)
5. **MSYS bash corrompe caminhos:** Usar r'C:\Users\dell-...' ao inves de '/c/Users/dell-...'

## Configuracoes Importantes
- **epr_client.json:** heartbeat=15s, sync_interval=30s, debounce=1000ms
- **epr_server.json:** porta 8443, SSL com certificado auto-gerado
- **EPR_SECRET:** variavel de ambiente no systemd da VPS
- **Cert fingerprint:** b198b7efa84bacde9941cb2b8f14e30f36fe218132076fdea49aa3d624ad6952

## Proximos Passos
1. Aguardar sincronizacao completa dos 618 pending (skills/templates)
2. Limpar __pycache__ do banco da VPS tambem
3. Configurar cliente EPR como servico persistente (Scheduled Task no Windows)
4. Adicionar metricas de sincronizacao ao dashboard
5. Teste de conflito: modificar arquivo simultaneamente em ambos os lados
