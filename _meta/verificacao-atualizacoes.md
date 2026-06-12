# Verificação de Atualizações — 2026-06-10

## Resumo
- **Data:** 2026-06-10 (cron job)
- **Script original:** NÃO ENCONTRADO (`verificar_atualizacoes.py` não existe)
- **Ação:** Verificação executada manualmente (pitfall 15.279)

## Hermes Agent
- **Instalado:** v0.14.0
- **Disponível:** v0.16.0 (tag: v2026.6.5, publicada 06/06/2026)
- **Status:** ⚠️ ATUALIZAÇÃO DISPONÍVEL (2 versões atrás)
- **Commits pendentes:** ~20+ commits entre HEAD local e origin/main

## Destaques da v0.16.0 (The Surface Release)
- fix(docker): otimização de tamanho de imagem
- fix(curator): atomic state writer compartilhado
- fix(desktop): esconder console children no Windows
- feat(tts): suporte a Gemini audio tag rewrite + persona prompt
- fix(memory,skills): reparo write-approval inline prompt, gateway staging, /skills review
- fix(update): self-heal de venv quebrada por instalação interrompida
- fix(config): preservar modo original do .env em remove_env_value
- fix(openrouter): route reasoning_effort para verbosity em modelos Anthropic
- fix(gateway): Slack approval UX em threads
- fix(cli): reparo stdout/stderr não-UTF-8 em todas as plataformas
- fix(desktop): sticky user bubbles no titlebar drag region
- Auto-restart gateway após Telegram QR onboarding

## Recomendação
Atualizar para v0.16.0. Correções relevantes para o ambiente Windows:
- Desktop console children fix (relatado em sessões anteriores)
- OpenRouter reasoning_effort fix
- Memory/skills write-approval fix
- CLI non-UTF-8 stdout fix

## Comando sugerido
```
pip install --upgrade hermes-agent
```
