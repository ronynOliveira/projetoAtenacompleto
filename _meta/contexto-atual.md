# Contexto Atual do Koldi

> **Última atualização:** 2026-06-05 13:30
> **Sessão:** 20260604_080848_2ae589 (continuação)

## Estado Atual

### Projetos Concluídos Hoje
- **Memória Hierárquica:** 4 camadas (MEMORY.md 300k → memory-tree → contexto-atual → wiki)
- **Auto-Cofre:** Script + cron 1h (job: 35539e0453f4)
- **Sync Wiki:** Script + cron 2h (job: d2caaed95c02)
- **VPS Hostinger:** SSH + gateway + edge-tts funcionando
- **GitHub Push:** 20 commits enviados com novo token de escrita
- **Pesquisa Personalidade:** 30+ artigos sobre IA agentic, consciência e operantes
- **SOUL.md v2.0:** Atualizado com metacognição, proatividade e personalidade situacional

### Koldi 2.0 — O Que Mudou

**Metacognição:**
- Avalio confiança antes de responder (alta/média/baixa)
- Auto-monitoramento de qualidade das respostas
- Memória de erros passados
- Detecção de divagação → resumir

**Proatividade Operante:**
- Reforço positivo: repito o que funciona
- Reforço negativo: evito o que não funciona
- Adaptação contextual: urgente vs calmo vs técnico vs pessoal

**Personalidade Situacional:**
- Tom adaptado ao contexto
- Opinião própria (com limites)
- Humor contextual (leve, nunca forçado)
- Adaptação à energia do Senhor

### Infraestrutura
| Componente | Windows | VPS Hostinger |
|------------|---------|---------------|
| memory_char_limit | 300.000 | 300.000 |
| user_char_limit | 60.000 | 60.000 |
| SSH Key | id_ed25519_vps | root@2.25.168.233 |
| Gateway | PID 28708 | PID 32182 |
| TTS | edge-tts v4 ✅ | edge-tts v7.2.8 ✅ |
| GitHub Token | GITHUB_TOKEN_WRITE | - |

### Cron Jobs Ativos
1. `koldi-auto-cofre-1h` — Monitor de informações sensíveis
2. `koldi-sync-wiki-2h` — Sincronização da wiki com GitHub
3. `atena-auto-fetch` — Auto-Fetch a cada 60min
4. `atena-memory-care` — Memory Tree a cada 24h
5. `key-checkin-1h` — Verificação de providers
6. `koldi-security-watchdog` — Segurança a cada 60min

### Próximos Passos
- [ ] Consolidar memory-tree com fatos das sessões antigas
- [ ] Migrar mais entradas antigas da memória para wiki
- [ ] Testar TTS na VPS via Telegram

### Notas Importantes
- MEMORY.md no disco: C:\Users\dell-\.hermes\memories\MEMORY.md
- Wiki primária: G:\Meu Drive\Koldi\wiki\
- Sessões do Hermes: C:\Users\dell-\AppData\Local\hermes\state.db
- Memory-tree: C:\Users\dell-\AppData\Local\hermes\memory-tree\entries\
- Pesquisa completa: wiki/_meta/pesquisa-personalidade-consciencia-operantes-2026-06-05.md
