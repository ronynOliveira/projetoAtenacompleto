# Corrigir subprocess para não abrir janelas CMD no Windows

## Problema
Quando scripts Python usam `subprocess.run()` ou `subprocess.Popen()` no Windows sem especificar `creationflags=subprocess.CREATE_NO_WINDOW`, uma janela CMD é aberta a cada chamada.

## Solução
Adicionar no topo do arquivo (após os imports):
```python
import subprocess
import sys
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
```

E em TODAS as chamadas `subprocess.run()` e `subprocess.Popen()`, adicionar:
```python
creationflags=_NO_WINDOW,
```

## Aplicado em
- `scripts/tts_koldi.py` — 6 chamadas corrigidas (já estava corrigido na versão do wiki)

## Descoberta relacionada
O `execute_code` do Hermes pode abrir janelas CMD quando roda scripts Python que usam subprocess sem `CREATE_NO_WINDOW`. Se o Senhor ver janelas CMD abrindo e fechando, verificar:
1. Se há scripts maliciosos no Startup (como `StartApp2.ps1` que foi removido hoje)
2. Se scripts nossos usam subprocess sem a flag

## Bandeiras alternativas
- `subprocess.CREATE_NO_WINDOW` — mais direto, recomendado
- `STARTF_USESHOWWINDOW` — via `startupinfo`, mais verboso mas compatível com versões antigas
- `stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL` — para silenciar completamente
