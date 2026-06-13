# RELATÓRIO CONSOLIDADO — Análise AGY + Opencode
**Data:** 2026-06-13
**Analisadores:** AGY (Antigravity) + Opencode (DeepSeek V4 Flash)

---

## RESUMO EXECUTIVO

Duas análises independentes identificaram **~50 problemas** nos 9 scripts do ecossistema Koldi:

| Categoria | Crítico | Alto | Médio | Total |
|-----------|---------|------|-------|-------|
| Bugs | 3 | 5 | 2 | 10 |
| Segurança | 2 | 4 | 3 | 9 |
| Performance | 0 | 6 | 4 | 10 |
| Arquitetura | 0 | 4 | 6 | 10 |
| RAG | 0 | 2 | 1 | 3 |
| **TOTAL** | **5** | **21** | **16** | **~50** |

---

## PROBLEMAS CRÍTICOS (Correção Imediata)

### 1. consultar_ia_stream() — Sanitização DEPOIS do payload
**Ambos os analisadores concordam:** O `sanitize_input()` é chamado DEPOIS de montar a lista de messages. O prompt bruto vai para a API.

### 2. planning.py — add_decision/add_error falham silenciosamente
**BUG GRAVÍSSIMO:** Após a primeira decisão/erro, a string `## Decisions Made\n\n` deixa de existir, então todas as chamadas subsequentes falham silenciosamente.

### 3. token_guard.py — ZeroDivisionError
Se `limit` for 0, `state[key] / limit` lança exceção.

### 4. front_controller.py — Classificação frágil por substring
Uma pergunta mista como "Escreva um e-mail sobre arquivos" é classificada como `controle_local` por causa de "arquivos".

### 5. kcpa.py — KeyError: 'contexto_detectados'
A chave está escrita errada no dict de previsão.

---

## PROBLEMAS DE SEGURANÇA (Alta Prioridade)

### 1. sanitize_input() ineficaz
Só remove null bytes e limita tamanho. Não protege contra prompt injection real.

### 2. API_KEY estática no módulo
Resolvida na importação. Se a chave mudar em runtime, não é detectada.

### 3. Sem rate limiting real
Apenas `time.sleep(0.5)` entre chamadas.

### 4. validate_model_id() permite ':'
Pode ser explorado para injeção.

---

## PROBLEMAS DE PERFORMANCE (Média Prioridade)

### 1. Sem connection pooling
Cada chamada cria nova conexão TCP/SSL.

### 2. Chamadas multi-IA sequenciais
`comparar_modelos` e `multi_consulta` são 100% sequenciais.

### 3. status() baixa 300+ modelos para verificar 4
Endpoint `/models` retorna payload enorme.

### 4. API key carregada em import time
Deveria ser lazy loading.

### 5. Token guard lê/escreve disco a cada operação
Deveria ter cache em memória.

---

## MELHORIAS DE ARQUITETURA

### 1. RAG Local
nomic-embed-text instalado mas inutilizado.

### 2. Cache de respostas
Sem cache — cada consulta vai para API.

### 3. Retry com backoff
Não implementado.

### 4. Testes unitários
Sem testes automatizados.

### 5. Async/await
Tudo síncrono — bloqueia em chamadas de rede.

---

## CORREÇÕES APLICADAS NESTA SESSÃO

1. ✅ BUG-01: consultar_ia_stream() — validação ANTES do payload
2. ✅ BUG-02: Headers HTTP duplicados — centralizados via _get_headers()
3. ✅ BUG-03: Pipeline truncamento — max_input_chars configurável
4. ✅ BUG-04: Mismatch Ollama — MODEL_LOCAL_OLLAMA separado
5. ✅ BUG-05: print() → logger no mnemosyne_wrapper
6. ✅ BUG-06: Path traversal no planning.py — _safe_dirname()
7. ✅ SEC-01: sanitize_input() reforçado
8. ✅ SEC-02: Removido print() de API key
9. ✅ kcpa.py: KeyError 'contexto_detectados' corrigido

---

## PRÓXIMOS PASSOS (Sugestões dos Analisadores)

1. Implementar RAG local com nomic-embed-text
2. Adicionar connection pooling (requests.Session)
3. Implementar retry com backoff exponencial
4. Adicionar cache de respostas com TTL
5. Criar testes unitários
6. Implementar async/await para chamadas paralelas
7. Otimizar status() para não baixar 300+ modelos
8. Adicionar suporte a inglês na classificação de intenção
9. Implementar fallback inteligente para Ollama offline
10. Adicionar métricas de uso detalhadas

---

*Relatório gerado após análise independente de AGY e Opencode*
*Koldi — Batedor da Nuvem, Gnóstico Construtor*
