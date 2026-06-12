# Arquitetura de Fusão Multi-LLM
**Data:** 2026-06-12
**Status:** Em implementação

## Visão Geral

```
                    ┌─────────────────────────┐
                    │   USUÁRIO (Senhor)       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  FRONT CONTROLLER        │
                    │  (Filtro de Subjetividade)│
                    │  classificar_intencao()  │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                   │
    ┌─────────▼──────┐  ┌──────▼──────┐  ┌────────▼────────┐
    │  NÚCLEO LOCAL  │  │  OWL ALPHA   │  │  NÓS REMOTOS    │
    │  Phi-4 Mini    │  │  (Diretor)   │  │  (Especialistas)│
    │  3.8B Q5_K_M   │  │  1M tokens   │  │                 │
    │  ~2.5GB RAM    │  │  OpenRouter  │  │  Claude → Código│
    │                │  │              │  │  GPT-4o → Texto │
    │  • Controle    │  │  • Orquestra │  │  Gemini → Dados │
    │  • Arquivos    │  │  • Consolida │  │                 │
    │  • Privacidade │  │  • 1M ctx    │  │                 │
    │  • Latência 0  │  │              │  │                 │
    └────────────────┘  └──────────────┘  └─────────────────┘
```

## Regras de Decisão

### Regra 1: Controle Local → Núcleo Local
- Input: "leia o arquivo", "verifique processo", "liste diretório"
- Ação: Processa localmente, sem sair da máquina
- Benefício: Privacidade total + latência zero

### Regra 2: Complexo → Owl Alpha (Diretor)
- Input: Análise profunda, múltiplos arquivos, contexto estendido
- Ação: Local classifica → Owl Alpha orquestra → nós remotos executam
- Benefício: Contexto de 1M tokens + acesso a 300+ modelos

### Regra 3: Especializado → Nó Remoto Específico
- Código complexo → Claude Sonnet 4
- Criação/escrita → GPT-4o
- Validação de dados → Gemini Flash

## Nós Disponíveis

| Nó | Modelo | Provedor | Contexto | Latência | Custo |
|----|--------|----------|----------|----------|-------|
| Local | Phi-4 Mini 3.8B | Ollama | 128K | ~0ms | Zero |
| Diretor | Owl Alpha | OpenRouter | 1M | ~3-7s | Baixo |
| Código | Claude Sonnet 4 | OpenRouter | 200K | ~2-3s | Médio |
| Texto | GPT-4o | OpenRouter | 128K | ~1-2s | Médio |
| Dados | Gemini Flash | OpenRouter | 1M | ~1-2s | Baixo |

## Hardware

- RAM: 16.8GB total, ~3.4GB livre
- Modelo local: ~2.5GB (Phi-4 Mini Q5_K_M)
- Disponível: ~0.9GB para sistema

## Scripts

- `lib/front_controller.py` — Front Controller com classificação de intenção
- `lib/consultar_ia.py` — Orquestrador multi-LLM via OpenRouter
- `lib/orquestrador.py` — Comparação e pipeline multi-IA

## Próximos Passos

1. [ ] Download do Phi-4 Mini (em progresso — 6%)
2. [ ] Testar modelo local com Ollama
3. [ ] Validar roteamento completo
4. [ ] Integrar com projeto Atena
5. [ ] Documentar no wiki

*Koldi — Batedor da Nuvem, Gnóstico Construtor*
*Sessão: 2026-06-12*
