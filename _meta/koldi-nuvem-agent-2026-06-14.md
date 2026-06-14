# Koldi Nuvem — Agente Remoto na VPS (2026-06-14)

## Arquitetura

```
┌─────────────────────┐         ┌─────────────────────┐
│   KOLDI LOCAL       │         │   KOLDI NUVEM       │
│   (Windows 10)      │◄─WSS──►│   (Debian VPS)      │
│                     │  :8443  │                     │
│  i5-1235U           │         │  1 core, 3.8GB      │
│  16GB RAM           │         │  48GB disco         │
│                     │         │                     │
│  EPR Client ────────┼────────►│ EPR Server          │
│  (PID 16220)        │         │ (PID 24460)         │
│                     │         │                     │
│  Local Agent        │  SSH    │ Nuvem Agent         │
│  (koldi_local_)     │  :22    │ (systemd, PID 25135)│
└─────────────────────┘         └─────────────────────┘
```

## Componentes

### EPR Bridge (WebSocket SSL)
- **Local:** epr_bridge.py --mode client (PID 16220)
- **VPS:** epr_bridge.py --mode server (PID 24460)
- **Protocolo:** HMAC + SSL pinning + cert fingerprint
- **Velocidade:** ~525 arquivos/min
- **Estado:** 1633 synced, 0 pending, 0 conflitos

### Koldi Nuvem Agent (VPS)
- **Serviço:** koldi-nuvem-agent.service (systemd, enabled)
- **PID:** 25135
- **Watch:** /opt/hermes/.hermes a cada 10s
- ** funções:** watchers de arquivos → handlers (wiki, python, scripts, memory, config, identity)
- **Logs:** /opt/hermes/logs/koldi_nuvem.log

### Controle Local (Windows)
- **Script:** scripts/koldi_nuvem_ctl.py
- **Funções:** status, start, stop, restart, log, stats, exec
- **Uso:** python koldi_nuvem_ctl.py status

### EPR Local Agent (Windows)
- **Script:** scripts/koldi_local_agent.py
- **Funções:** monitora EPR state DB → dispara tarefas na VPS
- **Uso:** python koldi_local_agent.py --watch

## Correções Aplicadas
1. Ortografia: "Nao" → "Não" (linha 371 do epr_bridge.py)
2. Status 'synced' atualizado após envio (bug crítico)
3. 473+ entradas __pycache__ removidas do banco
4. Deploy do código corrigido na VPS

## Próximos Passos
1. Adicionar mais handlers ao Nuvem Agent (testes, builds, deploys)
2. Configurar cron job local para manter EPR Client sempre rodando
3. Adicionar notificação via Telegram quando sync completar
4. Implementar retry automático para EPR Client (keepalive timeout ~2h)
5. Teste de conflito: modificar arquivo simultaneamente em ambos os lados
