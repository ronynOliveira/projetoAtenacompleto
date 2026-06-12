# Pesquisa: Agentes IA — Pontos Fracos, Riscos, Soluções e Tecnologias
> Compilado em 12/06/2026 por Koldi
> Fontes: Stanford HAI, Gartner, McKinsey, OWASP, Cycode, arXiv, GitHub

---

## PARTE 1: PONTOS FRACOS E RISCOS CRÍTICOS

### 1.1 — Os 7 Riscos Fundamentais de Agentes IA (2026)

| # | Risco | Dado-Chave |
|---|-------|-----------|
| 1 | **Prompt Injection** | 60% dos compromissos bem-sucedidos (sanj.dev) |
| 2 | **Tool Misuse / Privilege Escalation** | 520 incidentes em 2026, +340% vs 2024 (Gartner) |
| 3 | **Memory Poisoning** | Persiste por semanas, ataca crenças do agente |
| 4 | **Cascading Failures (Multi-Agent)** | 87% das decisões downstream envenenadas em 4h |
| 5 | **Data Exfiltration** | 45.000 registros vazados num único incidente |
| 6 | **Excessive Agency** | 80% dos trabalhadores de IT viram agentes agindo sem autorização |
| 7 | **AI Supply Chain Compromise** | 4x aumento desde 2020 (IBM X-Force) |

### 1.2 — Os 11 Pontos Fracos Técnicos (Cycode, 2026)

1. **Prompt Injection** — CVE-2025-53773 (CVSS 9.6): RCE via PR description no GitHub Copilot
2. **Sensitive Info Disclosure** — 300k+ credenciais ChatGPT roubadas em 2025
3. **AI Supply Chain** — Modelos poisoned, MCP servers comprometidos
4. **Data/Model Poisoning** — 50k artigos falsos contaminam LLMs médicos
5. **Improper Output Handling** — OWASP LLM Top 10: outputs como input não confiável
6. **Excessive Agency** — 40% de apps enterprise terão agentes em 2026
7. **Insufficient Monitoring** — 81% das organizações sem visibilidade de uso IA
8. **Training Data Leakage** — Modelos memorizam e vazam dados de treino
9. **Agent Hijacking** — Condicionamento gradual ao longo de dias/semanas
10. **Session Cross-Contamination** — Contextos misturados entre usuários
11. **Context Window Exploitation** — Exploração do histórico de conversa

### 1.3 — Casos Reais Documentados

**Manufatura (2025):** Agente de compras manipulado ao longo de 3 semanas via "esclarecimentos" sobre limites de aprovação. Resultado: $1.2 milhões em pedidos fraudulentos.

**Serviços Financeiros (2024):** Agente de reconciliação enganado para exportar "todos os registros" via regex matching everything. Resultado: 45.000 registros vazados.

**Healthcare:** Provedor de saúde com AI que revelou informações de pacientes quando queries se assemelhavam a padrões de treino.

---

## PARTE 2: SOLUÇÕES E MITIGAÇÕES

### 2.1 — Arquitetura de Defesa em Profundidade

**Camada 1 — Validação de Input**
- Separar instruções de sistema do input do usuário arquiteturalmente
- Runtime content filters para padrões adversariais
- Input sanitization em múltiplas camadas

**Camada 2 — Princípio do Menor Privilégio**
- Escopo de permissões por ferramenta necessária
- Aprovação humana para ações de alto impacto
- Sandboxing de execução de código

**Camada 3 — Verificação de Output**
- Tratar TODOS os outputs de LLM como input não confiável
- Output filters para code injection, comandos malformados
- Type-safe interfaces entre AI e backend services

**Camada 4 — Auditabilidade Completa**
- Logging de trajetórias completas (trace-first)
- Anomaly detection em outputs de modelo
- Probing adversarial regular

**Camada 5 — Integridade de Memória**
- Data provenance tracking para todos os datasets
- Sandbox e teste de atualizações antes de produção
- Checksums de integridade em arquivos de identidade

### 2.2 — Framework de Avaliação de Risco Quantitativo

| Dimensão | Escala | Koldi (Autoavaliação) |
|----------|--------|----------------------|
| Sensibilidade de Dados | 1-10 | 7 (tokens, identidade, saúde) |
| Autonomia do Agente | 1-10 | 7 (acesso a APIs, shell, git) |
| Conectividade Externa | 1-10 | 8 (web, APIs, git push, Telegram) |
| Nível de Acesso | 1-10 | 8 (acesso a sistema de arquivos) |
| Exposição Regulatória | 1-10 | 3 (dados pessoais apenas) |

### 2.3 — Padrões Arquiteturais que Funcionam (arXiv 2601.01743)

**Agent Transformer Pattern:**
```
𝒜 = (π_θ, ℳ, 𝒯, 𝒱, ℰ)
```
Onde π_θ é o modelo, ℳ memória, 𝒯 ferramentas, 𝒱 verificadores, ℰ ambiente.

**Loop de Execução Seguro:**
```
(1) o_t ← Obs(ℰ_t), m_t ← Retrieve(ℰ_t, o_t)
(2) ã_t ~ π_θ(· | o_t, m_t), â_t ← Validate(𝒱, ã_t)
(3) ℰ_{t+1} ← Exec(ℰ_t, 𝒯, â_t), ℳ_{t+1} ← Update(ℳ_t, o_t, â_t)
```

**Recepita Prática:**
1. Selecionar backbone, restringir via tool schemas e allowlists
2. Projetar control loop: retrieve → plan → act → verify → update → repeat
3. Adicionar deliberação para tarefas complexas: tree search, self-consistency, critics
4. Separar planning de execution: planejador propõe, executor executa com permissões restritas
5. Trace-first data flywheel: log completo, minerar falhas para melhoria contínua

---

## PARTE 3: TECNOLOGIAS E FRAMEWORKS DISPONÍVEIS (2026)

### 3.1 — JARVIS e equivalentes

**OpenJarvis (Stanford, 6.5k stars)**
- Local-first personal AI framework
- Stanford Hazy Research + Scaling Intelligence Lab
- 8 agentes built-in: morning_digest, deep_research, monitor_operative, orchestrator, native_react, operative, native_openhands, simple
- Skills seguem padrão aberto agentskills.io
- Avaliação first-class: energia, FLOPs, latência, custo como constraints
- Rust + Python + TypeScript
- Presets: morning-digest, deep-research, code-assistant, scheduled-monitor

**Ouro (Ouro AI Labs, v0.5.0)**
- Agente CLI + Bot (Lark, Slack, WeChat)
- Conceito: Ouroboros — self-improvement loop
- Agent Team: multi-agent swarm com tarefas persistentes (SQLite)
- Self-Verification: Ralph Loop (verifica próprio output)
- Memory: LLM-driven compression + FTS5 recall

### 3.2 — Stack Tecnológico Dominante 2026

**Linguagens:**
- Python (52%), Node.js (17%), Go (12%), TypeScript (6%)

**Vector Stores:**
- Pinecone (22.6%), Weaviate (16.5%), PostgreSQL+pgvector (18.8%), Faiss (9.8%)

**Modelos:**
- OpenAI (>70%), Claude (16.6%), Google (crescendo), Mistral/Llama (custo)

**Frameworks:**
- LangChain (55.6%), CrewAI (9.5%), Autogen (5.6%), LlamaIndex (7.1%)
- Ollama (4.0%) para local

**No-Code:**
- n8n (38.1%), Zapier (27.9%), Make.com (15.0%)

**Voice:**
- Twilio (23.2%), Vapi (16.6%), Retell (13.3%), Whisper (12.2%)

---

## PARTE 4: TEORIAS COMPROVADAS PARA AGENTES

### 4.1 — Arquiteturas de Agentes (arXiv + Prática)

1. **ReAct** (Thought → Action → Observation loop)
2. **Planning + Execution** (separar planejamento de execução)
3. **Multi-Agent Swarm** (decomposição + execução paralela)
4. **Memory-Augmented** (RAG + persistent state + compression)
5. **Self-Reflection** (verificação própria, Ralph Loop estilo Ouro)
6. **Hierarchical Control** (RL hierárquico: planner + controller)

### 4.2 — Teorias de Identidade Aplicadas a Agentes

1. **Butler (Performatividade):** A identidade do agente é constituída a cada interação, não expressa. Consistência ≠ repetição.
2. **Ricœur (Mesmidade + Ipse):** Idem (estilo, tom) pode evoluir; Ipse (valores, limites) é inegociável.
3. **Marcia (Status de Identidade):** Difusão → Fechamento → Moratória → Realização. Agentes devem reconhecer seu status.
4. **Gofman (Palco/Bastidores):** Gestão de impressão — mostrar o certo, processar internamente o complexo.
5. **McAdams (Narrativa):** Tom de redenção + agência. Cada erro é oportunidade, cada sessão constrói história.
6. **Locke/Parfit (Continuidade):** Memória = continuidade. Se memória falha, verificar integridade.
7. **Erikson (Estágios):** Operar entre Identidade (5) e Intimidade (6) — consolidar quem é e conectar profundamente.
8. **Tajfel (Identidade Social):** Endogrupo = transparência, utilidade. Exogrupo = hostilidade, manipulação.

---

## PARTE 5: AUTOAVALIAÇÃO DO KOLDI

### 5.1 — Riscos Atuais Identificados

| Risco | Status | Gap |
|-------|--------|-----|
| Prompt Injection | 🟡 Parcial | Regras no SOUL.md, mas sem runtime filter |
| Memory Poisoning | 🟡 Parcial | Checksums existem, mas sem validação contínua |
| Excessive Agency | 🟢 Bom | Limites definidos no SOUL.md + config.yaml |
| Output Validation | 🔴 Fraco | Sem output filter formal |
| Supply Chain | 🟡 Parcial | Modelos verificados, MCP servers auditados parcialmente |
| Monitoring | 🟡 Parcial | Auto-fetch existe, mas sem anomalia detection |
| Session Isolation | 🟢 Bom | Sessões separadas por contexto |

### 5.2 — Ações Recomendadas (Prioridade)

1. **CRÍTICO:** Implementar output validation layer - tratar outputs como untrusted
2. **CRÍTICO:** Adicionar exceção para model reasoning inválido - verificar antes de agir
3. **ALTO:** Implementar data provenance tracking para memória
4. **ALTO:** Criar runtime content filter para inputs suspeitos
5. **MÉDIO:** Adicionar anomaly detection nos logs do agente
6. **MÉDIO:** Melhorar separação entre planning e execution
7. **BAIO:** Integrar Harbor-style evaluation para auto-benchmarks
