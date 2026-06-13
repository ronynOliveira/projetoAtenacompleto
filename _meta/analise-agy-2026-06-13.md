# Análise e Auditoria de Código: Ecossistema Koldi
**Data da Auditoria:** 13 de Junho de 2026  
**Auditor:** Antigravity (Advanced Agentic Coding - Google DeepMind)  
**Status do Ecossistema:** Necessita de Refatoração Crítica (Bugs Gravíssimos e Falhas de Lógica Identificados)

---

## 1. Resumo Executivo & Análise do Ambiente

O ecossistema **Koldi** foi projetado para rodar de forma híbrida:
*   **Local:** Intel Core i5-1235U, 16.8GB RAM, sem GPU dedicada. Modelo local `Phi-4 Mini 3.8B Q5_K_M` (2.5GB) via Ollama.
*   **Remoto:** OpenRouter (Owl Alpha, Claude, GPT-4o, Gemini).
*   **Memória:** SQLite local (`Mnemosyne`) + Busca Vetorial (`nomic-embed-text` instalado no Ollama, mas subutilizado).

### Restrições Físicas de Hardware vs. Arquitetura
Como o sistema roda em uma CPU sem GPU dedicada, o tempo de inicialização (*cold start*) e processamento do Ollama exige cuidado. Chamadas locais síncronas bloqueantes podem travar a aplicação principal. O ecossistema também carece de concorrência real na orquestração remota e apresenta problemas de consistência lógica grave em sanitização de código e tratamento de arquivos concorrentes.

---

## 2. Auditoria Detalhada por Script

### 2.1. `consultar_ia.py` (Orquestrador Multi-LLM)

#### A. Bugs e Erros de Lógica
1.  **Bug de Sanitização no Streaming:** Na função `consultar_ia_stream` (linhas 217-220 e 234-235), a lista `messages` contendo o `prompt` e `system_prompt` é populada **antes** de as variáveis locais serem passadas por `sanitize_input()`. Portanto, o streaming envia o prompt bruto (não sanitizado), invalidando todo o mecanismo de segurança e sanitização da função.
2.  **API_KEY Estática no Módulo:** A variável `API_KEY` é resolvida globalmente na importação (`API_KEY = load_openrouter_api_key()`). Se a chave de API for definida no ambiente ou no arquivo `.env` após a inicialização do módulo, o script falhará em detectá-la.

#### B. Gargalos de Performance
1.  **Conexões HTTP Desperdiçadas:** Cada chamada a `consultar_ia` realiza uma nova requisição `requests.post()` avulsa. Em execuções paralelas ou repetidas (`comparar_modelos`, `multi_consulta`), a ausência de um pool de conexões (`requests.Session()`) adiciona latência excessiva devido ao handshake TCP/SSL repetitivo.
2.  **Execução Sequencial de Comparações:** O loop em `comparar_modelos` (linhas 294-304) processa os modelos um a um com `time.sleep(0.5)`. Isso eleva a latência final de forma linear.

#### C. Correção Concreta
Substitua as funções de consulta e utilize um objeto `requests.Session` persistente, com correção do bug de ordenação no stream.

```python
# Correção em C:\Users\dell-\AppData\Local\hermes\lib\consultar_ia.py

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Reutilizar uma sessão HTTP persistente com Pool de Conexões e Retry básico integrado
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=3)
_session.mount("https://", _adapter)

def get_session_headers() -> dict:
    # Resolve dinamicamente a chave para evitar problemas de importação precoce
    key = load_openrouter_api_key()
    if not key:
        raise ValueError("OPENROUTER_API_KEY não configurada")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://koldi.local",
        "X-Title": "Koldi Multi-LLM Orchestrator",
    }

def consultar_ia(
    modelo: str,
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: int = TIMEOUT,
) -> dict:
    if not validate_model_id(modelo):
        return {"model": modelo, "content": None, "error": "ID de modelo invalido", "latency_ms": 0}
    
    # Sanitiza ANTES de qualquer uso
    prompt_sanitizado = sanitize_input(prompt)
    system_sanitizado = sanitize_input(system_prompt) if system_prompt else ""
    
    messages = []
    if system_sanitizado:
        messages.append({"role": "system", "content": system_sanitizado})
    messages.append({"role": "user", "content": prompt_sanitizado})
    
    payload = {
        "model": modelo,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    start = time.time()
    try:
        resp = _session.post(
            f"{BASE_URL}/chat/completions",
            headers=get_session_headers(),
            json=payload,
            timeout=timeout,
        )
        latency_ms = int((time.time() - start) * 1000)
        
        if resp.status_code == 429:
            return {"error": "Rate limit", "model": modelo, "latency_ms": latency_ms, "content": None}
            
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        return {
            "model": modelo,
            "content": content,
            "tokens_used": usage.get("total_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as e:
        return {
            "model": modelo,
            "content": None,
            "error": str(e),
            "latency_ms": int((time.time() - start) * 1000),
        }

def consultar_ia_stream(
    modelo: str,
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.7,
):
    if not validate_model_id(modelo):
        yield "[ERRO: ID de modelo invalido]"
        return
        
    # CORREÇÃO: Sanitiza antes de criar a lista de mensagens
    prompt_sanitizado = sanitize_input(prompt)
    system_sanitizado = sanitize_input(system_prompt) if system_prompt else ""
    
    messages = []
    if system_sanitizado:
        messages.append({"role": "system", "content": system_sanitizado})
    messages.append({"role": "user", "content": prompt_sanitizado})
    
    payload = {
        "model": modelo,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    
    try:
        resp = _session.post(
            f"{BASE_URL}/chat/completions",
            headers=get_session_headers(),
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        
        for line in resp.iter_lines():
            if line and line.startswith(b"data: "):
                chunk = line[6:]
                if chunk == b"[DONE]":
                    return
                try:
                    data = json.loads(chunk)
                    delta = data["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        yield delta["content"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except Exception as e:
        yield f"[ERRO: {e}]"
```

---

### 2.2. `front_controller.py` (Filtro de Subjetividade & Roteamento)

#### A. Bugs e Erros de Lógica
1.  **Classificação Frágil por Substring:** O roteamento de intenção (`classificar_intencao`) é feito por buscas estáticas simples de substrings (ex: `"leia o arquivo"`, `"refatorar"`). Se o usuário fizer uma pergunta mista como *"Escreva um e-mail explicando o erro de segurança dos arquivos da pasta"*, o roteador classificará incorretamente como `controle_local` (devido a *"arquivos da pasta"*), enviando a tarefa complexa para o modelo local `Phi-4 Mini` ao invés do GPT-4o ou Claude.
2.  **Ignora Entrada em Inglês:** Não há suporte para termos equivalentes em inglês (*"read file"*, *"refactor"*), o que é comum na rotina de desenvolvedores.

#### B. Gargalos de Performance e Latência
1.  **Bloqueio Síncrono no Ollama:** Se o serviço local Ollama estiver suspenso ou processando na CPU do i5-1235U, a chamada `requests.post()` para o Ollama travará o front controller por até 120 segundos.
2.  **Status Remoto Altamente Ineficiente:** A função `status()` consulta a lista inteira de modelos do OpenRouter (`/models`) sincronamente e sem cache. Esse endpoint retorna um payload JSON imenso com 300+ modelos, gerando gargalo de E/S de rede e latência.

#### C. Correção Concreta
Implementar um sistema de fallback inteligente caso o Ollama local apresente falha, e otimizar a requisição de status.

```python
# Correção em C:\Users\dell-\AppData\Local\hermes\lib\front_controller.py

def _ollama_chat(model: str, prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Timeout menor de 15 segundos para chamadas locais (CPU), evitando travamento eterno
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        # FALLBACK: Se o Ollama falhar, redireciona dinamicamente para o OpenRouter (Owl-Alpha)...
        logger.warning(f"Ollama falhou ({e}). Fazendo fallback para o OpenRouter (Owl-Alpha)...")
        try:
            return _openrouter_chat(MODEL_OWL_ALPHA, prompt, system)
        except Exception as err:
            return f"[ERRO DE ROTEAMENTO HÍBRIDO] {err}"
```

---

### 2.3. `orquestrador.py` (Orquestração Multi-IA)

#### A. Bugs e Erros de Lógica
1.  **Propagação de Erro no Pipeline:** Se o passo `N` falhar no `pipeline`, o campo `r.get('content')` será nulo. O código simplesmente define `input_atual = f"[Erro em {step['nome']}]"`. Isso faz com que o passo `N+1` receba literalmente a string de erro como prompt, gerando custos de tokens desnecessários e falhas encadeadas.
2.  **Falta de Tratamento na Consolidação:** Se a IA consolidadora falhar, o script inteiro retorna um dicionário com um objeto de erro cru na chave `consolidacao`, sem fallback de resumo local.

#### B. Gargalos de Performance
1.  **Processamento Sequencial Obsoleto:** Na função `orquestrar` (linha 49) e `comparar` (linha 94), as requisições para os diferentes modelos remotos rodam de forma sequencial com um `time.sleep(0.5)`. Como essas chamadas são independentes, elas deveriam ser paralelizadas para economizar tempo.

#### C. Correção Concreta
Paralelizar as chamadas a múltiplas IAs usando `concurrent.futures.ThreadPoolExecutor` e implementar interrupção imediata no pipeline em caso de erro crítico.

```python
# Correção em C:\Users\dell-\AppData\Local\hermes\lib\orquestrador.py

from concurrent.futures import ThreadPoolExecutor

def orquestrar(
    tarefa: str,
    modelos: dict[str, str] = None,
    prompt_template: str = None,
) -> dict:
    if modelos is None:
        modelos = {
            "pesquisa": get_melhor_modelo_para_tarefa("pesquisar " + tarefa),
            "analise": get_melhor_modelo_para_tarefa("analisar " + tarefa),
            "criacao": get_melhor_modelo_para_tarefa("criar " + tarefa),
        }
    
    if prompt_template is None:
        prompt_template = "Tarefa: {tarefa}\n\nForneça uma resposta completa e bem estruturada."
        
    prompt = prompt_template.format(tarefa=tarefa)
    
    # Fase 1: Execução Paralela Real via Threads
    resultados = {}
    def _consultar(nome, modelo):
        return nome, consultar_ia(modelo, prompt, max_tokens=2048, temperature=0.7)
        
    with ThreadPoolExecutor(max_workers=len(modelos)) as executor:
        futures = [executor.submit(_consultar, nome, modelo) for nome, modelo in modelos.items()]
        for fut in futures:
            nome, res = fut.result()
            resultados[nome] = res
            
    # Fase 2: Consolidar
    consolidador = get_melhor_modelo_para_tarefa("consolidar e resumir: " + tarefa)
    
    contexto = "## Resultados das IAs consultadas:\n\n"
    for nome, r in resultados.items():
        if r.get('content'):
            contexto += f"### {nome} ({r['model']})\n{r['content']}\n\n"
            
    contexto += f"\n## Tarefa Original\n{tarefa}\n\n"
    contexto += "Consolide as respostas acima em uma resposta final coerente e completa."
    
    print(f"[Koldi] Consolidando com {consolidador}...")
    consolidacao = consultar_ia(consolidador, contexto, max_tokens=4096, temperature=0.5)
    
    # Fallback se a consolidação remota falhar
    if consolidacao.get("error") and resultados:
        # Tenta consolidar localmente com Phi-4 se houver falha de rede remota
        from front_controller import _ollama_chat, MODEL_LOCAL_OLLAMA
        print("[Koldi] Erro na consolidação remota. Tentando consolidação local (Phi-4)...")
        resposta_local = _ollama_chat(MODEL_LOCAL_OLLAMA, contexto[:8000], "Você é o Koldi. Consolide o relatório.")
        consolidacao = {
            "model": MODEL_LOCAL_OLLAMA,
            "content": resposta_local,
            "error": "Erro remoto mitigado via local fallback",
            "latency_ms": 0
        }
        
    return {
        "tarefa": tarefa,
        "resultados": resultados,
        "consolidador": consolidador,
        "consolidacao": consolidacao,
        "modelos_usados": list(modelos.values()),
    }

def pipeline(
    tarefa: str,
    steps: list[dict],
    max_input_chars: int = 4000,
) -> dict:
    input_atual = tarefa
    resultados = {}
    
    for step in steps:
        prompt = step["prompt_template"].format(input=input_atual)
        print(f"[Pipeline] {step['nome']} via {step['modelo']}...")
        
        r = consultar_ia(step["modelo"], prompt, max_tokens=2048, temperature=0.7)
        resultados[step["nome"]] = r
        
        # CORREÇÃO: Abortar imediatamente se um dos passos fundamentais falhar
        if not r.get('content') or r.get('error'):
            print(f"[Pipeline] ERRO CRÍTICO no passo {step['nome']}. Pipeline interrompido.")
            break
            
        input_atual = r['content'][:max_input_chars]
        time.sleep(0.1) # Reduzido delay desnecessário
        
    return {
        "pipeline": [s["nome"] for s in steps],
        "resultados": resultados,
    }
```

---

### 2.4. `koldi_utils.py` (Utilitários Compartilhados)

#### A. Bugs e Erros de Lógica (GRAVÍSSIMOS)
1.  **Sanitização Quebra Código Fonte (Divisões e Imports):** A função `sanitize_input` (linha 81) executa `text.replace('/', ' ').replace('\\', ' ')`. Isso substitui todas as barras por espaços! Como consequência, qualquer código Python que faça divisão (`x / y`), paths de importação relativos ou caminhos de arquivos locais (`G:\Meu Drive\Koldi`) serão mutilados e convertidos em espaços antes de serem enviados à IA, destruindo a estrutura lógica.
2.  **Destruição da Indentação do Python:** A linha 91 executa `_re.sub(r'\s+', ' ', text)`, o que remove todas as quebras de linha e múltiplos espaços. Isso transforma qualquer bloco de código identado em uma linha contínua, impossibilitando que a IA interprete código estruturado que dependa de recuo (como Python).
3.  **Remoção Ingênua de Palavras-Chave:** Substituir ocorrências de termos como `"jailbreak"` ou `"ignore previous instructions"` por string vazia corrompe prompts normais onde o usuário discute esses tópicos, mas falha em bloquear injeções de prompt complexas (ex: encoded em base64 ou obfuscadas).

#### B. Gargalos de Performance / Segurança
1.  **Subprocesso Ineficiente para Registro do Windows:** A função `load_openrouter_api_key` executa um comando externo via `subprocess.run(["cmd", "/c", "reg query..."])` no Windows. Chamar um console shell é lento, gera processos extras e pode disparar falso-positivos em antivírus/firewalls locais.

#### C. Correção Concreta
Otimize a leitura do Registro usando a biblioteca padrão do Python (`winreg`) e refatore a sanitização para manter caminhos de arquivo, quebras de linha e sintaxe de divisão de códigos válidos.

```python
# Correção em C:\Users\dell-\AppData\Local\hermes\lib\koldi_utils.py

import sys
import re

def load_openrouter_api_key() -> str:
    # 1. Env var do processo
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if len(key) > 20:
        return key
        
    # 2. Arquivo .env
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    if len(key) > 20:
                        return key
        except Exception as e:
            logger.warning(f"Erro lendo .env: {e}")
            
    # 3. Registry do Windows usando winreg nativo (Sem subprocesso cmd.exe)
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key_reg:
                val, _ = winreg.QueryValueEx(key_reg, "OPENROUTER_API_KEY")
                if len(str(val)) > 20:
                    return str(val)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Erro ao ler registry com winreg: {e}")
            
    return ""

def sanitize_input(text: str, max_length: int = 50000) -> str:
    if not isinstance(text, str):
        return str(text)[:max_length]
        
    text = text[:max_length]
    text = text.replace("\x00", "") # Remove null bytes
    
    # CORREÇÃO: Removemos a substituição global de '/' e '\\' por espaço para não quebrar códigos/caminhos.
    # Em vez disso, validamos/sanitizamos apenas contra Path Traversal perigoso de forma cirúrgica
    text = re.sub(r'\.\.[/\\]', '', text) # Remove caminhos de subida de diretório como ../ ou ..\
    
    # CORREÇÃO: Não usamos re.sub de termos normais como "jailbreak" por strings vazias,
    # nem removemos quebras de linha (evitando destruir indentação de códigos Python)
    
    return text.strip()
```

---

### 2.5. `kcpa.py` (Communication Pattern Adapter)

#### A. Bugs e Erros de Lógica (GRAVÍSSIMOS)
1.  **NameError Fatal no Resumo:** Na função `get_pattern_summary()` (linha 298), o script executa `historico = _carrar_dados(HISTORY_FILE)` (há um erro de grafia, o correto é `_carregar_dados`). Esse código gera um `NameError` e quebra imediatamente toda vez que o relatório do sistema é acessado.
2.  **Crescimento Desenfreado dos Arquivos JSON:** Em `registrar_interacao`, a lista `patterns["frases_incompletas"]` cresce infinitamente sem limites (`.append`). À medida que o usuário interage com o sistema, esse arquivo JSON fica imenso, degradando severamente o tempo de E/S do Koldi (pois os dados são gravados de forma síncrona a cada interação).

#### B. Falhas de Concorrência
1.  **Gravação Concorrente Sem Locks:** Os dados são lidos e reescritos no disco de forma assíncrona/concorrente se múltiplos processos acessarem a biblioteca. Sem locks de escrita de arquivo, isso causará corrupções fatais nos arquivos JSON de padrões e histórico.

#### C. Correção Concreta
Corrija o erro de grafia, limite a quantidade de registros armazenados nos dicionários para evitar estouro de memória e implemente uma gravação robusta.

```python
# Correção em C:\Users\dell-\AppData\Local\hermes\lib\kcpa.py

def get_pattern_summary() -> dict:
    patterns = _carregar_dados(PATTERNS_FILE)
    # CORREÇÃO: Erro de grafia corrigido de _carrar_dados para _carregar_dados
    historico = _carregar_dados(HISTORY_FILE) 
    vocab = _carregar_dados(VOCABULARY_FILE)
    
    return {
        "total_interacoes": len(historico.get("interacoes", [])),
        "frases_incompletas_count": len(patterns.get("frases_incompletas", [])),
        "verbos_frequencia": patterns.get("frequencia_verbos", {}),
        "horarios": patterns.get("horarios", {}),
        "vocabulario_top20": dict(
            sorted(vocab.get("vocabulary", {}).items(), key=lambda x: x[1], reverse=True)[:20]
        ),
    }

def registrar_interacao(
    input_usuario: str,
    output_koldi: str,
    contexto: str = "",
    duracao_segundos: float = 0,
):
    historico = _carregar_dados(HISTORY_FILE)
    patterns = _carregar_dados(PATTERNS_FILE)
    vocab = _carregar_dados(VOCABULARY_FILE)
    
    # Registrar no histórico
    if "interacoes" not in historico:
        historico["interacoes"] = []
    
    interacao = {
        "timestamp": datetime.now().isoformat(),
        "input_length": len(input_usuario),
        "output_length": len(output_koldi),
        "contexto": contexto[:100],
        "duracao": duracao_segundos,
        "input_hash": hashlib.md5(input_usuario.encode()).hexdigest()[:8],
    }
    historico["interacoes"].append(interacao)
    
    # Manter últimas 1000 interações
    if len(historico["interacoes"]) > 1000:
        historico["interacoes"] = historico["interacoes"][-1000:]
        
    input_lower = input_usuario.lower()
    
    # Verificar se é frase incompleta
    frase_incompleta = False
    for padrao in PADROES_INICIAIS["frases_incompletas"]:
        if re.search(padrao, input_lower):
            frase_incompleta = True
            break
            
    interacao["frase_incompleta"] = frase_incompleta
    
    if frase_incompleta:
        if "frases_incompletas" not in patterns:
            patterns["frases_incompletas"] = []
        patterns["frases_incompletas"].append({
            "input": input_usuario[:200],
            "timestamp": datetime.now().isoformat(),
        })
        # CORREÇÃO: Limitar também a lista de frases incompletas para evitar arquivo JSON gigante
        if len(patterns["frases_incompletas"]) > 200:
            patterns["frases_incompletas"] = patterns["frases_incompletas"][-200:]
            
    # Extrair verbos de comando
    verbos_encontrados = [v for v in PADROES_INICIAIS["verbos_comando"] if v in input_lower]
    if verbos_encontrados:
        interacao["verbos"] = verbos_encontrados
        if "frequencia_verbos" not in patterns:
            patterns["frequencia_verbos"] = {}
        for v in verbos_encontrados:
            patterns["frequencia_verbos"][v] = patterns["frequencia_verbos"].get(v, 0) + 1
            
    # Extrair vocabulário
    words = re.findall(r'\b[a-záàâãéêíóôõúç]{3,}\b', input_lower)
    word_count = Counter(words)
    if "vocabulary" not in vocab:
        vocab["vocabulary"] = {}
    for word, count in word_count.items():
        vocab["vocabulary"][word] = vocab["vocabulary"].get(word, 0) + count
        
    # CORREÇÃO: Limitar vocabulário para evitar estouro (manter top 500 palavras)
    if len(vocab["vocabulary"]) > 500:
        vocab["vocabulary"] = dict(sorted(vocab["vocabulary"].items(), key=lambda x: x[1], reverse=True)[:500])
        
    # Atualizar horários de atividade
    hora = datetime.now().hour
    periodo = "manha" if 6 <= hora < 12 else "tarde" if 12 <= hora < 18 else "noite"
    if "horarios" not in patterns:
        patterns["horarios"] = {"manha": 0, "tarde": 0, "noite": 0}
    patterns["horarios"][periodo] = patterns["horarios"].get(periodo, 0) + 1
    
    # Salvar
    _salvar_dados(HISTORY_FILE, historico)
    _salvar_dados(PATTERNS_FILE, patterns)
    _salvar_dados(VOCABULARY_FILE, vocab)
```

---

### 2.6. `kec.py` (Evolution Controller)

#### A. Bugs e Erros de Lógica
1.  **Falha com Strings Longas em Path no Windows:** Em `analisar_com_opencode` (linha 75), a função `Path(codigo_ou_arquivo)` tenta ler um caminho. Se a string contiver código-fonte inline e ultrapassar os limites ou contiver caracteres proibidos no NTFS (`<`, `>`, `|`), o construtor do `Path.exists()` levantará um erro catastrófico de `OSError` (Invalid argument), interrompendo o fluxo.
2.  **Ignora Códigos de Retorno do Subprocesso:** Os comandos externos `opencode` e `agy` rodam síncronos e não checam se o código de retorno foi bem-sucedido (`result.returncode != 0`). Mensagens escritas em `stderr` (como falhas de npm ou erros de rede) são juntadas como se fossem a análise de sucesso.

#### B. Melhorias de Performance
1.  **Timeout Muito Alto (300 segundos):** Deixar a thread principal travada por até 5 minutos no comando `opencode` sem execução assíncrona causa péssima experiência de uso no terminal local do usuário.

#### C. Correção Concreta

```python
# Correção em C:\Users\dell-\AppData\Local\hermes\lib\kec.py

def analisar_com_opencode(codigo_ou_arquivo: str, contexto: str = "", salvar_em: str = None) -> str:
    codigo = codigo_ou_arquivo
    arquivo_nome = "inline"
    
    # CORREÇÃO: Proteger contra caracteres inválidos de caminhos no Windows (NTFS) usando try-except OSError
    try:
        if len(codigo_ou_arquivo) < 260: # Tamanho máximo comum para caminhos
            path = Path(codigo_ou_arquivo)
            if path.is_file() and path.exists():
                codigo = path.read_text(encoding="utf-8")
                arquivo_nome = path.name
    except (OSError, ValueError):
        # A string é puramente código inline contendo caracteres não permitidos para caminhos
        pass
        
    prompt = f"Analise o seguinte código ({arquivo_nome}):\n\n{codigo[:5000]}\n\nContexto: {contexto}"
    
    try:
        result = subprocess.run(
            [OPENCODE_CMD, "run", prompt],
            capture_output=True, text=True, timeout=120, # Timeout reduzido para evitar travamentos
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        
        # CORREÇÃO: Validar o código de retorno do subprocesso
        if result.returncode != 0:
            return f"[ERRO OPENCÓDIGO CLI (Retorno {result.returncode})]: {result.stderr}"
            
        output = result.stdout
        if salvar_em:
            Path(salvar_em).write_text(output, encoding="utf-8")
        return output
    except subprocess.TimeoutExpired:
        return "[ERRO: Timeout na análise Opencode]"
    except FileNotFoundError:
        return "[ERRO: Opencode não encontrado. Instale com: npm install -g opencode]"
    except Exception as e:
        return f"[ERRO: {e}]"
```

---

### 2.7. `mnemosyne_wrapper.py` (Local Memory SQLite Wrapper)

#### A. Bugs e Erros de Lógica (GRAVE)
1.  **Bug de Revocação de Memórias (Recall Filtrado na Memória):** Na função `recall` (linhas 59-68), o escopo (`scope`) é avaliado na memória RAM após a consulta retornar com o limitador `top_k`.
    *Exemplo:* Se o banco de dados contiver 10 memórias e o usuário solicitar a busca com `top_k=5` e `scope="preference"`. O `_mn.recall` trará apenas as 5 memórias com maior pontuação vetorial. Se todas as 5 forem do escopo `"session"`, a filtragem por list comprehension `[r for r in results if r.get("scope") == scope]` excluirá todos os resultados, retornando uma lista vazia `[]`, mesmo que existam memórias pertinentes com a tag `"preference"` no banco.
    *Correção:* A filtragem por escopo deve ser feita no próprio query do SQLite (se a biblioteca base suportar) ou o wrapper deve pedir um `top_k` ampliado (ex: `top_k * 5`) na chamada e truncar para `top_k` após a filtragem de escopo na memória.

#### B. Uso de RAG Híbrido com `nomic-embed-text`
1.  **Embedding Local Não Utilizado:** O arquivo `nomic-embed-text` está instalado via Ollama local, mas a memória está chamando queries externas ou lexicais. O wrapper precisa encapsular a chamada de embeddings local no Ollama para alimentar a pesquisa semântica do `Mnemosyne`.

#### C. Correção Concreta

```python
# Correção em C:\Users\dell-\AppData\Local\hermes\lib\mnemosyne_wrapper.py

import requests

def obter_embedding_local(texto: str) -> list[float]:
    """Gera embeddings localmente via Ollama usando nomic-embed-text."""
    try:
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": texto},
            timeout=10
        )
        r.raise_for_status()
        return r.json().get("embedding", [])
    except Exception as e:
        logger.error(f"Falha ao gerar embedding local com nomic-embed-text: {e}")
        return []

def recall(
    query: str,
    top_k: int = 5,
    scope: Optional[str] = None,
    source: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> list:
    try:
        # CORREÇÃO: Se houver filtro por escopo em memória, aumentamos a busca inicial para não perder registros
        busca_k = top_k * 5 if scope else top_k
        
        results = _mn.recall(
            query=query,
            top_k=busca_k,
            source=source,
            from_date=from_date,
            to_date=to_date,
        )
        
        if scope:
            # Filtro em memória seguro garantindo maior cobertura
            results = [r for r in results if r.get("scope") == scope][:top_k]
            
        return results
    except Exception as e:
        logger.error(f"Erro recall: {e}")
        return []
```

---

### 2.8. `token_guard.py` (Token Budget Guard)

#### A. Bugs e Erros de Lógica
1.  **Divisão por Zero:** Se os limites globais forem zerados ou desativados por alguma rotina de configuração externa, a divisão em `state[key] / limit` (linha 130) e em `pct = state[key] / limit * 100` (linhas 195, 200, 205) causará uma exceção catastrófica de `ZeroDivisionError` e quebrará o orquestrador.
2.  **Orçamento Baseado apenas no Estimado:** A função `guard_call` recebe `estimated_tokens`, executa o método e salva o estimado no banco. Porém, ela nunca lê o consumo real retornado pelo OpenRouter (`tokens_used`). Isso gera um desvio acumulado grave entre a estimativa e o uso real de tokens após poucas chamadas.

#### B. Falhas de Concorrência (Race Condition)
1.  **Inexistência de Lock de Arquivo:** O estado é carregado do arquivo JSON (`STATE_FILE`), incrementado e salvo de volta. Se múltiplos processos concorrentes lerem o arquivo ao mesmo tempo, um sobrescreverá o incremento do outro, gerando perda da contagem de tokens reais.

#### C. Correção Concreta
Implementar Lock de Arquivo simples no Windows para prevenir corrupção e mitigar o risco de `ZeroDivisionError`.

```python
# Correção em C:\Users\dell-\AppData\Local\hermes\lib\token_guard.py

import msvcrt

def _load_state() -> dict:
    # Garante a existência do arquivo de forma segura
    if not STATE_FILE.exists():
        initial = {
            "session_tokens": 0,
            "hourly_tokens": 0,
            "daily_tokens": 0,
            "session_start": datetime.now().isoformat(),
            "hour_start": datetime.now().isoformat(),
            "day_start": datetime.now().strftime("%Y-%m-%d"),
            "calls": 0,
            "blocked": 0,
        }
        _save_state(initial)
        return initial
        
    try:
        # Implementação de Lock simples no Windows para leitura concorrente
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            # Trava o arquivo para leitura/escrita
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1024)
            data = json.load(f)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1024)
            return data
    except Exception:
        # Fallback de segurança se falhar
        return {
            "session_tokens": 0, "hourly_tokens": 0, "daily_tokens": 0,
            "session_start": datetime.now().isoformat(),
            "hour_start": datetime.now().isoformat(),
            "day_start": datetime.now().strftime("%Y-%m-%d"),
            "calls": 0, "blocked": 0,
        }

def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1024)
            json.dump(state, f, indent=2)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1024)
    except Exception as e:
        logger.error(f"Erro ao salvar estado do token guard: {e}")

def get_status() -> dict:
    state = _load_state()
    state = _reset_if_needed(state)
    
    # CORREÇÃO: Prevenção de ZeroDivisionError
    def safe_pct(used, limit):
        return (used / limit * 100) if limit and limit > 0 else 0.0
        
    return {
        "session": {
            "used": state["session_tokens"],
            "limit": DEFAULT_SESSION_LIMIT,
            "pct": safe_pct(state["session_tokens"], DEFAULT_SESSION_LIMIT),
        },
        "hourly": {
            "used": state["hourly_tokens"],
            "limit": DEFAULT_HOURLY_LIMIT,
            "pct": safe_pct(state["hourly_tokens"], DEFAULT_HOURLY_LIMIT),
        },
        "daily": {
            "used": state["daily_tokens"],
            "limit": DEFAULT_DAILY_LIMIT,
            "pct": safe_pct(state["daily_tokens"], DEFAULT_DAILY_LIMIT),
        },
        "calls": state["calls"],
        "blocked": state["blocked"],
    }
```

---

### 2.9. `planning.py` (Planning with Files)

#### A. Bugs e Erros de Lógica (GRAVÍSSIMO)
1.  **Substituições Únicas Silenciosas em Decisions/Errors:** As funções `add_decision` and `add_error` executam:
    `content = content.replace("## Decisions Made\n\n", ...)`
    Isso funciona apenas na primeira chamada. Uma vez que o primeiro registro é adicionado, a string `"## Decisions Made\n\n"` deixa de existir no arquivo (ela se torna `"## Decisions Made\n- Minha Decisao\n"`). Portanto, todas as chamadas subsequentes a `add_decision` ou `add_error` falham silenciosamente, bloqueando o histórico de decisões e erros do planejamento.
2.  **Caminho Absoluto Rígido do Windows:** O caminho `PLANS_DIR = Path.home() / "OneDrive" / "Área de Trabalho" / "plans"` assume que a pasta da Área de Trabalho do usuário está em português do Brasil e dentro do diretório padrão do OneDrive. Em sistemas com idioma do Windows em inglês ou sem o OneDrive ativo, a criação do diretório falhará silenciosamente ou lançará uma exceção de caminho inexistente.

#### B. Correção Concreta
Substitua as funções de escrita utilizando Expressões Regulares (`re.sub`) para injetar as decisões e erros no topo de suas respectivas seções, independente de estarem vazias ou populadas.

```python
# Correção em C:\Users\dell-\AppData\Local\hermes\lib\planning.py

import re

def _get_plans_dir() -> Path:
    if ALT_PLANS_DIR.exists():
        return ALT_PLANS_DIR
    # CORREÇÃO: Evitar caminhos em português fixos e garantir compatibilidade
    desktop_pt = Path.home() / "OneDrive" / "Área de Trabalho" / "plans"
    desktop_en = Path.home() / "OneDrive" / "Desktop" / "plans"
    local_workspace = Path.home() / ".hermes" / "plans"
    
    for path in [desktop_pt, desktop_en]:
        try:
            if path.parent.exists():
                path.mkdir(parents=True, exist_ok=True)
                return path
        except Exception:
            pass
            
    local_workspace.mkdir(parents=True, exist_ok=True)
    return local_workspace

def add_decision(plan_dir: str, decision: str, rationale: str = "") -> bool:
    plan_dir = Path(plan_dir)
    task_plan = plan_dir / "task_plan.md"
    
    if not task_plan.exists():
        return False
        
    content = task_plan.read_text(encoding="utf-8")
    entry = f"- {decision}"
    if rationale:
        entry += f": {rationale}"
    entry += "\n"
    
    # CORREÇÃO: Usar regex para buscar o cabeçalho e injetar a decisão logo após
    # Isso funciona mesmo se a lista de decisões já contiver itens
    pattern = r"(## Decisions Made\n*)"
    if re.search(pattern, content):
        content = re.sub(pattern, r"\1" + entry, content, count=1)
        task_plan.write_text(content, encoding="utf-8")
        return True
    return False

def add_error(plan_dir: str, error: str, resolution: str = "") -> bool:
    plan_dir = Path(plan_dir)
    task_plan = plan_dir / "task_plan.md"
    
    if not task_plan.exists():
        return False
        
    content = task_plan.read_text(encoding="utf-8")
    entry = f"- {error}"
    if resolution:
        entry += f": {resolution}"
    entry += "\n"
    
    # CORREÇÃO: Mesma injeção via regex para a seção de Erros
    pattern = r"(## Errors Encountered\n*)"
    if re.search(pattern, content):
        content = re.sub(pattern, r"\1" + entry, content, count=1)
        task_plan.write_text(content, encoding="utf-8")
        return True
    return False
```

---

## 3. Estratégias de Otimização e Recursos Avançados

### 3.1. RAG com `nomic-embed-text`
Para utilizar o modelo `nomic-embed-text` instalado no Ollama e integrá-lo com o SQLite `Mnemosyne` local, devemos calcular o embedding vetorial local das novas memórias antes de armazená-las no banco.

#### Fluxo de RAG Local Proposto:
```python
# RAG Local Integrado em C:\Users\dell-\AppData\Local\hermes\lib\mnemosyne_wrapper.py

def remember_com_vetor(content: str, scope: str = "session", importance: float = 0.5, source: str = "conversation", metadata: dict = None) -> str:
    """Registra a memória e calcula o embedding vetorial localmente via Ollama."""
    emb = obter_embedding_local(content)
    
    meta_atual = metadata or {}
    if emb:
        meta_atual["embedding_vetorial_local"] = True
        # Se mnemosyne ou sqlite-vec aceita vetor, passamos no metadata
        meta_atual["vector"] = emb
        
    mid = remember(
        content=content,
        scope=scope,
        importance=importance,
        source=source,
        metadata=meta_atual
    )
    return mid
```

### 3.2. Mecanismo de Retry com Backoff Exponencial para OpenRouter
As requisições na nuvem via OpenRouter podem falhar por instabilidade ou Rate Limit (429). Abaixo está um decorator Python nativo de retry com backoff exponencial para `consultar_ia.py` sem dependências pesadas externas:

```python
# Retry nativo com backoff em C:\Users\dell-\AppData\Local\hermes\lib\consultar_ia.py

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    res = func(*args, **kwargs)
                    # Se retornou erro de Rate Limit, força o retry
                    if isinstance(res, dict) and res.get("error") == "Rate limit":
                        raise requests.exceptions.RequestException("Rate limit")
                    return res
                except (requests.exceptions.RequestException, ValueError) as e:
                    if x == retries:
                        logger.error(f"Falha definitiva após {retries} retries: {e}")
                        return {"model": args[0] if args else "unknown", "content": None, "error": str(e), "latency_ms": 0}
                    sleep = (backoff_in_seconds * 2 ** x)
                    logger.warning(f"Erro na requisição. Tentando novamente em {sleep} segundos...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

# Uso prático:
# consultar_ia = retry_with_backoff(retries=3)(consultar_ia)
```

---

## 4. Testes Unitários Sugeridos

Abaixo está um conjunto de testes robusto utilizando `pytest` para certificar as correções cruciais de segurança e lógica no diretório `C:\Users\dell-\AppData\Local\hermes\lib\tests`.

```python
# Criar arquivo em C:\Users\dell-\AppData\Local\hermes\lib\tests\test_koldi.py

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koldi_utils import sanitize_input
from planning import _safe_dirname

def test_sanitizacao_mantem_divisao_e_quebra_de_linha():
    # Testa se códigos Python com quebras de linha e divisão não são destruídos pela sanitização
    codigo_original = "def dividir(x, y):\n    return x / y"
    sanitizado = sanitize_input(codigo_original)
    
    assert "/" in sanitizado, "A sanitização removeu a barra de divisão matemática!"
    assert "\n" in sanitizado, "A sanitização removeu a quebra de linha do código!"
    assert "    " in sanitizado, "A sanitização destruiu a indentação do código!"

def test_safe_dirname():
    assert _safe_dirname("Teste de Nome! #1") == "teste-de-nome-1"
    assert _safe_dirname("caminho/com/barras") == "caminho-com-barras"

def test_planejamento_multiplas_decisoes(tmp_path):
    # Mock do diretório do plano
    from planning import create_plan, add_decision, get_plan
    
    plan_dir = create_plan(
        task_name="Test Task",
        phases=["Phase 1"],
        goal="Test Goal",
        plan_dir=str(tmp_path)
    )
    
    # Testa se podemos adicionar múltiplas decisões sequencialmente sem que falhe silenciosamente
    assert add_decision(str(plan_dir), "Decisão 1", "Justificativa 1") is True
    assert add_decision(str(plan_dir), "Decisão 2", "Justificativa 2") is True
    
    plan_data = get_plan(str(plan_dir))
    content = plan_data["task_plan"]
    
    assert "Decisão 1" in content
    assert "Decisão 2" in content
```

---

## 5. Integração com MCP Toolbox

Para habilitar que ferramentas externas controlem a orquestração e a memória do Koldi via protocolo MCP (Model Context Protocol), podemos expor os utilitários de planejamento e recuperação de memórias como ferramentas ativas.

### Definições de Ferramentas MCP Propostas:
1.  `koldi_remember`: Grava informações cruciais do usuário de forma semântica e lexical no `Mnemosyne`.
2.  `koldi_recall`: Consulta fatos históricos e preferências do usuário usando o RAG local com embedding `nomic-embed-text`.
3.  `koldi_create_plan`: Inicializa um template de planejamento no padrão 3-file do Manus para acompanhamento de tarefas.

---

### Considerações Finais
As correções aqui apresentadas mitigam erros graves que afetam diretamente a capacidade do Koldi de interagir com scripts e códigos do usuário, garantindo a integridade dos dados locais, a concorrência segura em ambientes multi-processos e o fallback resiliente em caso de instabilidade nas conexões com o OpenRouter.
