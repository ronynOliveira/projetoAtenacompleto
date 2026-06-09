# Diagnóstico e Correções — 2026-06-09

## Problemas Encontrados

### 1. Ollama Local — Reasoning Models
- **Problema:** Todos os modelos Ollama (gemma4:e2b, gemma4:e4b, gemma4:latest, qwen3:8b, deepseek-r1:8b) são reasoning models
- **Sintoma:** Retornam `content` vazio e tudo no campo `reasoning` que o Hermes não lê
- **Único modelo funcional:** `hermes3:8b`
- **Correção:** Trocado default no config.yaml de `gemma4:e2b` para `hermes3:8b`

### 2. OpenRouter — Chave Inválida
- **Problema:** Chave no cofre retorna HTTP 401 (expirada/revogada)
- **Causa:** Senhor usou bastante tokens tentando fazer funcionar

### 3. security_watchdog.py — SyntaxError
- **Problema:** Regex com escape tripla `\\\\s` na linha 76
- **Correção:** Corrigido para `\\s`

### 4. Cron Jobs com Connection Error
- `key-checkin-1h` — Connection error
- `koldi-auto-cofre-1h` — script auto_cofre.py não existe mais

### 5. VPS Hostinger
- **IP:** 2.25.168.233
- **Atualização:** Hermes Agent v0.15.x → v0.16.0 (2026.6.5)
- **Gateway:** Reiniciado e funcionando
- **Ollama:** Não instalado (VPS tem apenas 3.8GB RAM)

## Pesquisa: Rotação Automática de Chaves API

### Endpoint OpenRouter para Monitorar Créditos
```
GET /api/v1/key
Authorization: Bearer sk-or-...

Response:
{
  "data": {
    "label": "...",
    "limit": 100.0,           // limite total (null = ilimitado)
    "usage": 12.5,            // quanto já foi usado
    "limit_remaining": 87.5,  // créditos restantes
    "is_free_tier": false,
    "rate_limit": {...}
  }
}
```

### Headers de Rate Limit
- `x-ratelimit-limit` — limite total
- `x-ratelimit-remaining` — requisições restantes
- `x-ratelimit-reset` — timestamp de reset

### Estratégias de Rotação
1. **Dual-key pattern** — Criar chave nova enquanto a antiga ainda vive
2. **Proxy com fallback** — Round-robin entre N chaves com backoff em 429
3. **Management API** — `POST /keys` → deploy → `DELETE /keys/{hash}`
4. **BYOK** — Chaves do provider ficam na conta; só rotaciona acesso

### Script Criado
- `scripts/openrouter_key_monitor.py` — Monitor completo com:
  - Verificação de créditos via `/api/v1/key`
  - Alerta via Telegram quando créditos < threshold
  - Fallback automático entre múltiplas chaves
  - Cache para não exceder rate limits
  - Cron job a cada 6h

### Referências
- [OpenRouter API Key Rotation docs](https://openrouter.ai/docs/cookbook/administration/api-key-rotation)
- [Management API Reference](https://openrouter.ai/docs/guides/overview/auth/management-api-keys)
- [openrouter-proxy (rotação automática)](https://github.com/AxionAura/openrouter-proxy)
