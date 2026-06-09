# Monitor de Sistema — Projeto Atena

## Última verificação: 2026-06-09 12:54

| Recurso | Valor | Status |
|---------|-------|--------|
| CPU | 13% | ✅ Saudável |
| RAM | 64.5% usado (5.6GB livre / 15.7GB total) | ✅ Saudável |
| Disco C: | 57.9% usado (392.4GB livre / 933GB) | ✅ Saudável |
| Disco G: | 60% usado (372.8GB livre / 933GB) | ✅ Saudável |
| Rede | ✅ OK (ping 8.8.8.8: 8ms média) | ✅ Saudável |

## Notas
- Script `monitor_sistema.py` tem bug de parsing com vírgula decimal (locale BR). Valores de RAM/disco do script não são confiáveis.
- Valores reais obtidos via PowerShell direto.
- Nenhum alerta crítico. Sistema saudável.
