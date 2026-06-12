# Relatório: 4 Novas Ferramentas Implementadas
**Data:** 2026-06-12
**Sessão:** Implementação de ferramentas da lista de 20 do Senhor Robério

---

## Resumo Executivo

Das 20 ferramentas analisadas, 4 foram implementadas como complemento ao ecossistema existente:
1. **Mnemosyne** — memória local ultrarrápida (SQLite)
2. **MCP Toolbox for Databases** — 29 ferramentas SQL para Postgres da VPS
3. **Tokencap** — proteção contra loops de tokens
4. **Planning with Files** — padrão Manus de planejamento persistente

**Não implementadas (redundantes com o que já temos):**
- AgentMemory → já temos Mem0 v2 + Memory Tree + wiki
- MCP-Mem0 → já temos Mem0 v2 direto
- MCP Servers (Google, Composio) → já temos Composio MCP + Google Workspace MCP
- Google Search MCP → já temos ddgs + Kimi WebBridge
- WebSearch-MCP → precisa de crawler externo, temos alternativas

---

## 1. MNEMOSYE — Memória Local Ultrarrápida

### O que é
Sistema de memória zero-dependência para agentes IA. Usa SQLite + sqlite-vec + FTS5. Sub-millisecond retrieval.

### O que Koldi pode fazer
- **Armazenar memória instantaneamente**: `remember("preferência", scope="preference", importance=0.9)` → retorna ID em <1ms
- **Recall semântico**: `recall("preferências do usuário")` → retorna memórias relevantes com score
- **Scopes separados**: session, global, preference, health, config, system
- **Importância ponderada**: memórias mais importantes aparecem primeiro
- **CLI completo**: `mnemosyne store/recall/stats/export/import/backup`
- **MCP Server nativo**: `mnemosyne mcp --transport stdio` → 23 ferramentas MCP

### Como complementa o que já tínhamos
| Ferramenta anterior | Mnemosyne (novo) | Relação |
|---------------------|-------------------|---------|
| Mem0 v2 (Postgres externo, busca vetorial) | SQLite local, zero config | **Complementar**: Mem0 para busca vetorial profunda, Mnemosyne para lookup instantâneo local |
| Memory Tree (Python, scoring) | SQLite + FTS5 + vec | **Complementar**: Memory Tree para gerenciamento, Mnemosyne para armazenamento |
| Wiki (markdown, busca manual) | Busca semântica automática | **Complementar**: Wiki para conhecimento estruturado, Mnemosyne para fatos rápidos |

### Arquivos criados
- `lib/mnemosyne_wrapper.py` — wrapper Python com remember/recall/get_stats
- `lib/mnemosyne_mcp_config.py` — configuração MCP
- 12 memórias iniciais armazenadas (perfil, saúde, preferências, config)

---

## 2. MCP TOOLBOX FOR DATABASES — 29 Ferramentas SQL

### O que é
Servidor MCP oficial do Google que expõe ferramentas de banco de dados. Suporta 29+ ferramentas prontas para Postgres, MySQL, BigQuery, etc.

### O que Koldi pode fazer (na VPS)
- **`execute_sql`**: executar queries SQL arbitrárias no Postgres da VPS
- **`list_tables`**: listar todas as tabelas com esquema completo
- **`database_overview`**: overview do banco (tamanho, conexões, etc.)
- **`list_indexes`**: listar índices de tabelas
- **`list_views`**, **`list_schemas`**, **`list_sequences`**: introspeção completa
- **`list_active_queries`**, **`list_locks`**, **`long_running_transactions`**: monitoramento
- **`list_top_bloated_tables`**, **`list_table_stats`**: otimização
- **`get_query_plan`**: explain de queries
- **`replication_stats`**, **`list_replication_slots`**: replicação
- **`list_pg_settings`**, **`list_memory_configurations`**: configuração

### Como complementa o que já tínhamos
| Ferramenta anterior | MCP Toolbox (novo) | Relação |
|---------------------|---------------------|---------|
| psql manual via SSH | Tools SQL via MCP | **Complementar**: Toolbox para queries estruturadas, psql para administração |
| Mem0 (memórias vetoriais) | SQL direto nos checkpoints | **Complementar**: Mem0 para busca semântica, SQL para queries analíticas |
| Unison (sync de arquivos) | Acesso direto ao Postgres | **Complementar**: Unison para sync, Toolbox para dados |

### Bancos acessíveis
- **koldi_checkpoints**: checkpoints de sessão, memórias vetoriais Mem0, blobs, writes

### Arquivos criados
- `lib/toolbox_pg.py` — wrapper Python com sql_query/list_tables/db_stats
- `/opt/hermes/.hermes/tools/toolbox-mcp.sh` — script de inicialização
- `/opt/hermes/.hermes/tools/toolbox-pg.env` — credenciais

---

## 3. TOKENCAP — Proteção contra Loops de Tokens

### O que é
Biblioteca Python para rastrear uso de tokens e enforce budgets. Previne loops infinitos que geram faturas astronômicas.

### O que Koldi pode fazer
- **`check_budget(estimated_tokens)`**: verifica se há budget disponível antes de executar
- **`record_usage(tokens)`**: registra uso real de tokens
- **`guard_call(fn, limit)`**: executa função somente se houver budget
- **`get_status()`**: retorna uso atual (session/hourly/daily)

### Limites configurados
| Período | Limite | Quando reseta |
|---------|--------|---------------|
| Session | 100.000 tokens | Nova sessão |
| Hourly | 500.000 tokens | A cada hora |
| Daily | 2.000.000 tokens | Meia-noite |

### Alertas
- **80% usage**: warning automático
- **100% usage**: bloqueia execução, incrementa contador de blocked

### Como complementa o que já tínhamos
| Ferramenta anterior | Tokencap (novo) | Relação |
|---------------------|------------------|---------|
| TokenJuice (compressão de texto) | Budget enforcement | **Complementar**: TokenJuice comprime, Tokencap bloqueia loops |
| rate_limit_handler.py (backoff HTTP) | Controle de tokens | **Complementar**: rate_limit para API, Tokencap para orçamento |
| Cron jobs (automáticos) | Proteção para cron | **Complementar**: Cron dispara, Tokencap protege |

### Onde usar
- Cron jobs que fazem muitas chamadas de API
- Scripts de auto-evolução
- delegate_task com subagentes múltiplos
- Qualquer processo que rode autonomamente

---

## 4. PLANNING WITH FILES — Padrão Manus ($2B)

### O que é
Padrão de workflow do Manus (adquirido por $2B). Usa 3 arquivos markdown como "working memory on disk":
1. **`task_plan.md`** — fases, progresso, decisões, erros
2. **`notes.md`** — pesquisa, findings, fontes
3. **`[deliverable].md`** — output final

### O que Koldi pode fazer
- **`create_plan(name, phases)`**: cria estrutura completa de plano
- **`update_phase(plan, phase_idx)`**: marca fases como completas
- **`add_note(plan, text)`**: salva findings durante pesquisa
- **`add_decision(plan, decision, rationale)`**: registra decisões
- **`add_error(plan, error, resolution)`**: log de erros
- **`get_plan(plan)`**: lê estado do plano para continuar

### Como complementa o que já tínhamos
| Ferramenta anterior | Planning with Files (novo) | Relação |
|---------------------|----------------------------|---------|
| SOUL.md (valores, modos) | Task plans (execução) | **Complementar**: SOUL para identidade, Planning para execução |
| Wiki (_meta/sessao-*.md) | Notes.md (findings) | **Complementar**: Wiki para memória permanente, Notes para pesquisa temporária |
| Todo list (session) | Phases (checkpoints) | **Complementar**: Todo para tarefas simples, Planning para projetos complexos |
| plan skill (Hermes) | 3-file pattern | **Complementar**: plan skill para ideias, Planning para execução estruturada |

### Diretório de planos
- Local: `~/OneDrive/Área de Trabalho/plans/`
- Alternativo: `G:\Meu Drive\Koldi\wiki\_meta\plans\`

---

## MAPA COMPLEMENTARIDADE — VISÃO GERAL

```
ANTES (ecossistema existente):
├── Memória: Mem0 v2 + Memory Tree + Wiki + MEMORY.md
├── Busca: ddgs + Kimisearch + Kimi WebBridge + web_search
├── MCP: Composio MCP + Google Workspace MCP + CodeGraph MCP
├── Tokens: TokenJuice + rate_limit_handler + config.yaml otimizado
└── Planejamento: todo list + plan skill + SOUL.md modos

DEPOIS (com novas ferramentas):
├── Memória: + Mnemosyne (local, sub-ms)
├── SQL: + MCP Toolbox (29 tools Postgres)
├── Proteção: + Tokencap (budget enforcement)
└── Planejamento: + Planning with Files (Manus 3-file pattern)

TUDO CONECTADO:
Mnemosyne ←busca local→ Mem0 ←sql→ MCP Toolbox
     ↓                           ↓
  Wiki ←notes→ Planning ←guard→ Tokencap
```

---

## FERRAMENTAS NÃO IMPLEMENTADAS (e por quê)

| Ferramenta | Motivo |
|-----------|--------|
| AgentMemory (21.9k stars) | Redundante com Mem0 v2 + Memory Tree |
| MCP-Mem0 | Já temos Mem0 v2 direto |
| A-MEM MCP (33 stars) | Interessante mas nicho, Zettelkasten já coberto pela wiki |
| Google Search MCP | Precisa API key, temos ddgs + Kimi |
| WebSearch-MCP | Precisa crawler Docker, complexidade desnecessária |
| Google PSE MCP | Precisa Google CSE setup, temos web_search |
| Google Search Console | Só útil para SEO, não é nosso foco |
| Hermes Desktop Companion | Interface gráfica,參議院 no CLI |

---

## ESTATÍSTICAS DA SESSÃO

- **4 ferramentas implementadas**
- **3 scripts Python criados** (mnemosyne_wrapper, toolbox_pg, planning)
- **1 skill criada** (planning-with-files)
- **12 memórias Mnemosyne armazenadas**
- **29 ferramentas SQL disponíveis** via MCP Toolbox
- **1 MCP server configurado** na VPS (toolbox-postgres)

---

*Koldi — Batedor da Nuvem, Gnóstico Construtor*
*Sessão: 2026-06-12*
