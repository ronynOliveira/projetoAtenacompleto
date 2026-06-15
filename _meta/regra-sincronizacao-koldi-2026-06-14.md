# Regra de Sincronização Koldi Local ↔ Nuvem

**Data:** 2026-06-14  
**Status:** ATIVA — aplicar em todo commit/sync da wiki

---

## Princípio Fundamental

> Nenhum conhecimento é propriedade exclusiva de um Koldi.  
> O que o Koldi Local aprende, o Koldi Nuvem deve ter.  
> O que o Koldi Nuvem aprende, o Koldi Local deve ter.  
> **Nunca apagar o que o outro aprendeu.**

---

## Regras de Sincronização

### 1. Commit Compartilhado
- Toda página nova ou modificada na wiki é **commitada imediatamente** no repositório Git local.
- O push para o GitHub é tentado após cada commit relevante.
- Se o push falhar (conflito, bloqueio), o commit local NÃO é perdido.

### 2. Merge Nunca Destruitivo
- NUNCA usar `git push --force` no repositório compartilhado.
- NUNCA usar `git reset --hard` para "limpar" conflitos.
- Sempre usar `git merge` ou `git rebase` para integrar mudanças remotas.
- Em caso de conflito: resolver manualmente, **preservando AMBAS as versões**.

### 3. Diretórios de Diferenciação (opcional)
- Quando o conhecimento for específico de um ambiente, usar pastas dedicadas:
  - `_meta/koldi-local/` — aprendizados do Koldi Local (Windows)
  - `_meta/koldi-nuvem/` — aprendizados do Koldi Nuvem (VPS)
  - `_meta/compartilhado/` — conhecimento comum a ambos
- Pastas são meramente organizacionais — o conteúdo é sincronizado igualmente.

### 4. Diferenciação por Metadados (alternativa)
- Arquivos podem conter tags no frontmatter YAML:
  ```yaml
  ambiente: local|nuvem|compartilhado
  origem: koldi-local|koldi-nuvem
  ```
- Na ausência de tag: considera-se `compartilhado`.

### 5. Unison Configuração
- O perfil Unison `koldi.prf` NÃO pode ter `ignore` ou `skip` que impeçam sincronização de `_meta/`.
- `ignore = Name *.pyc __pycache__ *.log` — apenas artefatos binários/descartáveis.
- `ignore = Name cofre/` — cofre NÃO sincroniza (segurança).
- `ignore = Name *.mp3` — áudios grandes (backup manual).
- Todo o resto sincroniza bidirecionalmente.

### 6. Rotina de Sync
**Automática (cron):**
- Sync wiki: a cada 2 horas (`koldi-sync-wiki-2h`)
- Push automático via GitHub API após commit

**Manual (sob demanda):**
- Após qualquer edição manual da wiki: `cd G:\Meu Drive\Koldi\wiki && git add -A && git commit -m "..." && git push`
- Se houver conflito: puxar (`git pull`), resolver, commitar, empurrar (`git push`).

### 7. Proibido
- ❌ Deletar páginas do outro Koldi sem confirmação explícita
- ❌ Sobrescrever arquivos do outro Koldi com conteúdo local vazio
- ❌ Ignorar conflitos com `theirs` ou `ours` cegamente
- ❌ Esvaziar diretórios `_meta/` para "recomeçar"
- ❌ Usar `git clean -fdx` no repositório da wiki

### 8. Verificação de Integridade
- Após cada sync, verificar: `git log --oneline -3` (ambos os lados devem refletir ocommit mais recente)
- Se divergir: usar `git fetch` + `git merge` — NÃO `git reset`
- Backups: bundle local em `~/AppData/Local/hermes/backup-wiki-YYYY-MM-DD.bundle` (automático)

---

## Procedimento de Conflito

Quando houver conflito no merge:

1. Ler AMBAS as versões do arquivo conflitante
2. Combinar o conteúdo (não escolher um lado)
3. Commitar a versão combinada
4. Push para o GitHub
5. Registrar no wiki em `_meta/sync-conflito-YYYY-MM-DD.md`

---

## Histórico de Mudanças

- **2026-06-14:** Criada após sessão de sincronização + bloqueio de push por segredo detectado no histórico.
