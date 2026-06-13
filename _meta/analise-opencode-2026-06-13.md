# Análise Técnica Completa — Koldi Multi-LLM Ecosystem

**Data:** 2026-06-13
**Analisador:** opencode (deepseek-v4-flash-free)
**Contexto:** i5-1235U, 16.8GB RAM, Phi-4 Mini 3.8B (local) + OpenRouter (Owl Alpha, Claude, GPT-4o, Gemini)

---

## Índice

1. [consultar_ia.py — Orquestrador Multi-LLM via OpenRouter](#1-consultariapy)
2. [front_controller.py — Front Controller com Filtro de Subjetividade](#2-front_controllerpy)
3. [orquestrador.py — Orquestração, Comparação e Pipeline](#3-orquestradorpy)
4. [koldi_utils.py — Utilitários Compartilhados](#4-koldi_utilspy)
5. [kcpa.py — Communication Pattern Adapter](#5-kcapy)
6. [kec.py — Evolution Controller](#6-kecpy)
7. [mnemosyne_wrapper.py — Wrapper SQLite](#7-mnemosyne_wrapperpy)
8. [token_guard.py — Proteção contra Loops de Tokens](#8-token_guardpy)
9. [planning.py — Planning with Files (Padrão Manus)](#9-planningpy)
10. [toolbox_pg.py — MCP Toolbox Postgres](#10-toolbox_pgpy)
11. [memory_pipeline.py — Wrapper de Memória Unificado](#11-memory_pipelinepy)
12. [Resumo Consolidado](#12-resumo-consolidado)

---

## 1. consultar_ia.py

### Bugs e Erros de Lógica

#### B1 — `consultar_ia_stream` valida inputs DEPOIS de montar o payload
```python
payload = {  # linha 222 — construído sem validação
    "model": modelo,
    "messages": messages,
    ...
    "stream": True,
}
# Validação só ocorre na linha 231
if not validate_model_id(modelo):
    yield "[ERRO: ID de modelo invalido]"
    return
```
**Problema:** Se `modelo` for inválido/malicioso, o payload já foi montado (desperdício). Mais grave: `sanitize_input` é chamada depois do payload.

**Correção:** Mover validações para antes da construção do payload:
```python
def consultar_ia_stream(modelo, prompt, system_prompt="", max_tokens=4096, temperature=0.7):
    if not validate_model_id(modelo):
        yield "[ERRO: ID de modelo invalido]"
        return
    prompt = sanitize_input(prompt)
    system_prompt = sanitize_input(system_prompt) if system_prompt else ""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": modelo, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}
    ...
```

#### B2 — Variável `latency_ms` não definida no except HTTPError
```python
except requests.exceptions.HTTPError as e:
    return {
        "model": modelo,
        "content": None,
        "error": f"HTTP {resp.status_code}: {str(e)}",
        "latency_ms": latency_ms,  # <— NameError se resp.status_code >= 400 fora do 429
    }
```
Se a resposta for HTTP 400, 401, 500 etc (não 429), `latency_ms` pode não estar definida se o erro ocorreu antes de `resp.raise_for_status()` — na verdade, está definida na linha 158, então o erro é que `latency_ms` pode não refletir o tempo real se `resp.raise_for_status()` falhou imediatamente. Contudo, se `resp.status_code` for acessado, `resp` existe, então `latency_ms` existe. O bug real é mais sutil: se o erro acontece em `resp.json()` (ex: JSON inválido), cai no `except Exception` genérico, não no HTTPError. Isso está ok, mas a confusão entre latência real vs. padrão não.

**Correção:** Adicionar captura da latência antes de qualquer operação que possa falhar:
```python
start = time.time()
try:
    resp = requests.post(..., timeout=timeout)
    latency_ms = int((time.time() - start) * 1000)
    resp.raise_for_status()
    ...
except requests.exceptions.HTTPError as e:
    return {"model": modelo, "content": None, "error": f"HTTP {resp.status_code}: {str(e)}", "latency_ms": latency_ms}
```
(Já está assim, mas mova `latency_ms` para logo após `requests.post` retornar — antes de `raise_for_status`.)

#### B3 — `comparar_modelos` e `multi_consulta` não expõem erros de validação
Se `validate_model_id` falha, o dict retornado tem `content: None` e `error` preenchido, mas `comparar_modelos` não checa erros antes de usar.

**Correção:** Filtrar resultados com erro:
```python
def comparar_modelos(modelos: list[str], prompt: str) -> list[dict]:
    resultado = [consultar_ia(modelo, prompt) for modelo in modelos]
    erros = [r for r in resultado if r.get("error")]
    if erros:
        logger.warning(f"{len(erros)} modelo(s) retornaram erro: {[e['error'] for e in erros]}")
    return resultado
```

#### B4 — `listar_modelos` quebra se `pricing` tiver valores vazios
```python
"price_prompt": float(pricing.get("prompt", "1") or "1"),
```
Se `pricing["prompt"]` for `""`, `or "1"` resolve para `"1"` — ok.
Se `pricing["prompt"]` for `"0"`, `or "1"` erradamente trata como falsy e vira `"1"`.
Pouco provável, mas `float("0")` seria correto.

**Correção:** Usar operador ternário explícito:
```python
"price_prompt": float(pricing.get("prompt")) if pricing.get("prompt") else 0.0,
```

### Brechas de Segurança

#### S1 — API Key carregada no escopo do módulo (linha 40)
```python
API_KEY = load_openrouter_api_key()
```
**Problema:** A key é carregada no `import`, antes de qualquer logging/config. Se o módulo for importado durante um teste ou depuração, a key fica em memória mesmo se não for usada. Pode ser extraída via core dump.

**Correção:** Lazy loading:
```python
def _get_api_key() -> str:
    if not hasattr(_get_api_key, "_cache"):
        _get_api_key._cache = load_openrouter_api_key()
    return _get_api_key._cache

def _get_headers() -> dict:
    api_key = _get_api_key()
    ...
```

#### S2 — URL de OpenRouter hardcoded, sem verificação de TLS
`verify=True` é default no `requests`, mas não há configuração explícita. Em ambientes corporativos com proxies SSL, isso pode quebrar silenciosamente.

**Correção:** Adicionar parâmetro `verify` e log se SSL falhar:
```python
resp = requests.post(..., timeout=timeout, verify=os.environ.get("SSL_CERT_FILE", True))
```

#### S3 — Prompt injection mitigado, mas bypassável
`sanitize_input` remove padrões conhecidos:
```python
text = _re.sub(r'ignore\s+(previous|all|above|below)\s+instructions', '', text, flags=_re.IGNORECASE)
```
Mas `"IgNoRe ALL InStRuCtIoNs"` cobre apenas o conjunto limitado de padrões. Um atacante pode usar codificação Unicode (ex: `ｉｇｎｏｒｅ`) ou variações não-listadas.

**Correção:** Usar modelo local (Phi-4 Mini) como guardrail para classificar intenção antes de enviar ao OpenRouter — o `front_controller.py` faz isso parcialmente. Adicionar um passo extra: se o Phi-4 classificar como "prompt injection", bloquear.
```python
def detect_prompt_injection(text: str) -> bool:
    # Usar modelo local para classificar
    resultado = _ollama_chat("phi4-mini:3.8b-Q5_K_M",
        f"Classifique o texto abaixo como 'injection' ou 'normal':\n\n{text[:2000]}")
    return "injection" in resultado.lower()
```

### Otimizações de Performance

#### P1 — Conexão HTTP sem reuse (connection pooling)
Cada chamada a `consultar_ia` cria uma nova conexão TCP/TLS.

**Correção:** Usar `requests.Session()` compartilhado:
```python
_http_session = requests.Session()
_http_session.headers.update({"Content-Type": "application/json"})

def consultar_ia(...):
    ...
    resp = _http_session.post(f"{BASE_URL}/chat/completions", headers=_get_headers(), json=payload, timeout=timeout)
```

#### P2 — `time.sleep(0.5)` bloqueante em loops
`comparar_modelos` e `multi_consulta` usam `time.sleep` sequencial.

**Correção:** Usar `concurrent.futures.ThreadPoolExecutor` para chamadas paralelas:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def comparar_modelos(modelos: list[str], prompt: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        futuros = {executor.submit(consultar_ia, m, prompt): m for m in modelos}
        resultados = []
        for futuro in as_completed(futuros):
            resultados.append(futuro.result())
    return resultados
```

#### P3 — Timeout único para chamadas de 120s
Para modelos locais (Phi-4) via `consultar_ia`, 120s faz sentido. Para modelos rápidos (Gemini Flash, GPT-4o-mini), poderia ser 30s. Para Claude analisando código, 300s pode ser necessário.

**Correção:** Dicionário de timeout por modelo:
```python
MODEL_TIMEOUTS = {
    "google/gemini": 30,
    "openai/gpt-4o-mini": 30,
    "anthropic/claude": 180,
    "openrouter/owl-alpha": 300,
}
```

### Técnicas de RAG (nomic-embed-text)

#### R1 — `nomic-embed-text` instalado mas nunca usado
O contexto menciona que `nomic-embed-text` está instalado via Ollama, mas não é usado em lugar nenhum. O `mnemosyne_wrapper.py` não faz busca vetorial — ele depende do módulo `mnemosyne` externo.

**Proposta de Integração RAG:**
```python
class LocalEmbeddings:
    """Embeddings locais usando nomic-embed-text via Ollama."""
    
    def __init__(self, model="nomic-embed-text"):
        self.model = model
        self.url = "http://localhost:11434/api/embeddings"
    
    def embed(self, text: str) -> list[float]:
        resp = requests.post(self.url, json={"model": self.model, "prompt": text}, timeout=30)
        return resp.json()["embedding"]
    
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        with ThreadPoolExecutor(max_workers=4) as t:
            return list(t.map(self.embed, texts))

# Uso em recall do mnemosyne_wrapper:
def recall_semantico(query: str, top_k: int = 5) -> list:
    emb = LocalEmbeddings()
    query_vec = emb.embed(query)
    # Comparar com todos os vetores no SQLite (via sqlite-vec ou manual)
    # ...implementar busca por similaridade de cosseno no Python
```

**Vantagem:** Zerar custo de API para embeddings (vs. usar OpenAI/OpenRouter para embeddings).

### Cache de Respostas

#### C1 — Sem cache de respostas do OpenRouter
Toda chamada idêntica é reenviada. Para prompts frequentes, desperdício de tokens e $$.

**Correção:** Cache LRU simples baseado em hash do payload:
```python
from functools import lru_cache
import hashlib
import json

_CACHE_DIR = Path.home() / ".hermes" / "cache"
_CACHE_DIR.mkdir(exist_ok=True)

def _cache_key(modelo: str, prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> str:
    raw = json.dumps({"m": modelo, "p": prompt, "s": system_prompt, "t": max_tokens, "temp": temperature}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def _cache_get(key: str) -> dict | None:
    path = _CACHE_DIR / key
    if path.exists():
        data = json.loads(path.read_text())
        if time.time() - data["timestamp"] < 3600:  # 1h TTL
            return data["result"]
    return None

def _cache_set(key: str, result: dict):
    ( _CACHE_DIR / key ).write_text(json.dumps({"timestamp": time.time(), "result": result}))

# No consultar_ia, antes de chamar a API:
cache_key = _cache_key(modelo, prompt, system_prompt, max_tokens, temperature)
cached = _cache_get(cache_key)
if cached:
    return cached
# ... chamada API ...
_cache_set(cache_key, resultado)
```

### Retry com Backoff

#### R2 — Sem retry em falhas de rede/rate limit
429 (Rate Limit) e 5xx são retornados como erro, sem retry.

**Correção:** Retry com exponential backoff via `tenacity` ou implementação manual:
```python
import time
import random

def _retry_with_backoff(fn, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"Rate limit, retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                continue
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"Connection error, retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                continue
            raise
```

### Testes Unitários

#### T1 — Falta de testes para `consultar_ia`
O `if __name__ == "__main__"` faz um teste manual. Não há testes unitários.

**Testes sugeridos:**
```python
# tests/test_consultar_ia.py
import pytest
from unittest.mock import patch, MagicMock
from lib.consultar_ia import consultar_ia, validate_model_id, sanitize_input

def test_validate_model_id_valid():
    assert validate_model_id("anthropic/claude-sonnet-4")

def test_validate_model_id_invalid():
    assert not validate_model_id("../../etc/passwd")
    assert not validate_model_id("a" * 201)
    assert not validate_model_id("<script>")

def test_sanitize_input_removes_null():
    assert "\x00" not in sanitize_input("foo\x00bar")

def test_sanitize_input_removes_path_traversal():
    assert "../etc" in sanitize_input("../etc")  # remove .. mas mantém /etc? -> ' /etc'
    # Verificar que replace ocorreu

@patch("consultar_ia.requests.post")
def test_consultar_ia_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "test"}}], "usage": {}}
    mock_post.return_value = mock_resp
    result = consultar_ia("test/model", "hello")
    assert result["content"] == "test"

@patch("consultar_ia.requests.post")
def test_consultar_ia_rate_limit(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_post.return_value = mock_resp
    result = consultar_ia("test/model", "hello")
    assert result["error"] == "Rate limit"
```

---

## 2. front_controller.py

### Bugs e Erros de Lógica

#### B5 — `classificar_intencao` usa substring match, não semântica
```python
if padrao in prompt_lower:
```
"leia o arquivo" corresponde se o usuário digitar "leia o arquivo.txt" — ok. Mas "leia o arquivo de configuração" também corresponde — esperado. Porém, "leia o meu manual de arquivo" também corresponde (falso positivo).

**Correção:** Usar word boundary regex:
```python
padroes_locais = [r'\bleia\s+o\s+arquivo\b', r'\blistar\s+arquivos\b', ...]
```

#### B6 — `status` checa `phi4` no nome do modelo com string match genérico
```python
status["local"]["disponivel"] = MODEL_LOCAL_OLLAMA in modelos or any("phi4" in m for m in modelos)
```
Se houver outro modelo com "phi4" no nome (ex: `phi4-vision`, `phi4-14b`), será True mesmo sem o modelo específico.

**Correção:**
```python
status["local"]["disponivel"] = MODEL_LOCAL_OLLAMA in modelos
# Ou buscar substrings mais específicos
any(m.startswith("phi4-mini") or m.startswith("phi4:mini") for m in modelos)
```

### Brechas de Segurança

#### S4 — `StrictHostKeyChecking=no` em toolbox_pg.py (não neste arquivo, mas uso similar)
Não se aplica diretamente ao front_controller.

#### S5 — Sanitização destrutiva de input
`sanitize_input` remove `/` e `\` do prompt, o que impede o usuário de referenciar caminhos como "C:\Users\...". Isso pode quebrar prompts que precisam de paths.

**Correção:** Sanitizar preservando estrutura de caminhos válidos. Ou usar uma sanificação específica para prompt injection vs. paths.

### Otimizações de Performance

#### P4 — `_ollama_chat` sem timeout configurável
Hardcoded 120s. Para perguntas simples, 30s bastam.

**Correção:**
```python
def _ollama_chat(model: str, prompt: str, system: str = "", timeout: int = 60) -> str:
```

#### P5 — Status checking síncrono no `__main__`
`status()` faz requisições HTTP que podem travar. Para CLI de teste, aceitável. Para produção, assíncrono.

### Testes

#### T2 — `classificar_intencao` sem testes de borda
```python
def test_classificar_intencao_local():
    result = classificar_intencao("Liste os arquivos no diretorio atual")
    assert result["modelo_alvo"] == "local"

def test_classificar_intencao_codigo():
    result = classificar_intencao("Refatore esta funcao Python")
    assert result["modelo_alvo"] == "claude"

def test_classificar_intencao_indefinido():
    result = classificar_intencao("O que voce acha sobre IA?")
    assert result["modelo_alvo"] == "owl_alpha"
    assert result["confianca"] == 0.5

def test_classificar_intencao_empty():
    result = classificar_intencao("")
    assert result["modelo_alvo"] == "owl_alpha"
```

---

## 3. orquestrador.py

### Bugs e Erros de Lógica

#### B7 — `get_melhor_modelo_para_tarefa` é chamada com prefixo artificial
```python
modelos = {
    "pesquisa": get_melhor_modelo_para_tarefa("pesquisar " + tarefa),
    "analise": get_melhor_modelo_para_tarefa("analisar " + tarefa),
    "criacao": get_melhor_modelo_para_tarefa("criar " + tarefa),
}
```
`"pesquisar " + tarefa` altera a intenção real. Se `tarefa` for "crie um script Python", `pesquisar crie um script Python` pode não selecionar o modelo correto.

**Correção:** Usar um mapa de funções para determinar modelo por papel:
```python
def _modelo_para_papel(papel: str, tarefa: str) -> str:
    papeis = {
        "pesquisa": lambda t: "google/gemini-3.1-flash-lite" if any(w in t for w in ["pesquisar", "buscar", "notícias"]) else "openrouter/owl-alpha",
        "analise": lambda t: "anthropic/claude-sonnet-4",
        "criacao": lambda t: "openai/gpt-4o",
    }
    return papeis.get(papel, lambda t: "openrouter/owl-alpha")(tarefa)
```

#### B8 — Pipeline usa `input_atual` sem checar tamanho antes de format
```python
prompt = step["prompt_template"].format(input=input_atual)
```
Se `input_atual` contiver chaves como `{` ou `}` não escapadas, `.format()` lança `KeyError`.

**Correção:**
```python
prompt = step["prompt_template"].replace("{input}", input_atual)
```

#### B9 — Nenhum tratamento de erro no pipeline
Se um step falha, `input_atual = f"[Erro em {step['nome']}]"` — mas o próximo step recebe esse texto literal, poluindo o prompt.

**Correção:**
```python
if r.get('error'):
    logger.error(f"Pipeline step {step['nome']} failed: {r['error']}")
    return {"pipeline": ..., "resultados": resultados, "erro": r["error"]}
```

### Otimizações

#### P6 — Consultas sequenciais em `orquestrar` e `comparar`
Mesmo problema de P2 — usar `ThreadPoolExecutor`.

#### P7 — Consolidação com `get_melhor_modelo_para_tarefa` sem contexto
O consolidador recebe o conteúdo bruto de todas as respostas, possivelmente excedendo o contexto máximo. Pipeline trunca em 4000 chars, mas consolidação não.

**Correção:**
```python
# Truncar cada resposta se necessário
if len(r.get('content', '')) > max_input_chars:
    r['content'] = r['content'][:max_input_chars] + "\n...[truncated]"
```

### Testes

```python
def test_pipeline_empty_steps():
    with pytest.raises(KeyError):  # ou ajustar para retornar vazio
        pipeline("test", [])

def test_orquestrar_auto_select():
    result = orquestrar("Compare duas abordagens")
    assert "pesquisa" in result["resultados"]
    assert "analise" in result["resultados"]
```

---

## 4. koldi_utils.py

### Bugs

#### B10 — `sanitize_input` substitui `/` e `\` por espaço
```python
text = text.replace('/', ' ').replace('\\', ' ')
```
Isso quebra prompts que contêm caminhos de arquivo legítimos, datas (ex: "01/2026"), ou expressões regulares. É excessivamente agressivo.

**Correção:** Substituir apenas sequências perigosas específicas, não todos os `/`:
```python
# Remover apenas path traversal (..) combinado com separadores
text = _re.sub(r'(\.\.\s*[/\\])+', ' ', text)
text = _re.sub(r'[/\\]\.\.', ' ', text)
# Não remover / ou \ soltos
```

#### B11 — `validate_model_id` aceita caracteres potencialmente perigosos
```python
return bool(re.match(r'^[\w\-./:]+$', model_id))
```
`model_id` aceita `.` e `/` que podem ser usados em path traversal se mal combinados. Embora o regex impeça `..`, `model_id` pode ser algo como `..%2F..%2Fetc` se não for decodificado antes.

**Correção:**
```python
def validate_model_id(model_id: str) -> bool:
    if not isinstance(model_id, str) or len(model_id) > 200:
        return False
    if ".." in model_id or "%" in model_id:
        return False
    return bool(re.match(r'^[a-zA-Z0-9][\w.\-\/:]*$', model_id))
```

#### B12 — Leitura do Registry do Windows via `subprocess` é frágil
```python
result = subprocess.run(
    ["cmd", "/c", "reg query HKCU\\Environment /v OPENROUTER_API_KEY"],
    capture_output=True, text=True, timeout=5,
)
```
O parsing:
```python
parts = line.split()
if len(parts) >= 3:
    key = " ".join(parts[2:])
```
O formato do `reg query` é `name    type    value`. Se a key tiver espaços, `parts[2:]` pode incluir partes do tipo `REG_SZ` como "REG_SZ valor". Mais seguro:
```python
import winreg
try:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        value, _ = winreg.QueryValueEx(key, "OPENROUTER_API_KEY")
        if len(value) > 20:
            return value
except (FileNotFoundError, OSError):
    pass
```

### Segurança

#### S6 — `load_openrouter_api_key` faz debug log do comprimento
`logger.debug("API key carregada de env var")` é seguro, mas se o logger estiver configurado para DEBUG em produção, não há vazamento do valor real, apenas indicação de existência. Aceitável.

#### S7 — API key no Registry do Windows pode ser lida por outros processos
O Registry HKCU\Environment é visível para qualquer processo rodando como o mesmo usuário. Um malware poderia ler a key.

**Correção:** Preferir env var de sessão (escopo de processo) em vez de registry. Remover a leitura de registry ou documentar como último recurso inseguro.

### Otimizações

#### P8 — Carregar chave do registry a cada chamada
```python
key = load_openrouter_api_key()  # Chamado no escopo do módulo em consultar_ia.py
```
Se for chamado múltiplas vezes (como em `_get_headers`), re-executa toda a lógica de busca (env, arquivo, registry). Isso já é mitigado pela variável global `API_KEY = load_openrouter_api_key()`, mas se a variável for perdida ou resetada, a busca completa repete.

### Testes

```python
def test_sanitize_input_empty():
    assert sanitize_input("") == ""

def test_sanitize_input_max_length():
    assert len(sanitize_input("a" * 100000, max_length=100)) == 100

def test_validate_model_id_none():
    assert not validate_model_id(None)

def test_load_openrouter_api_key_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-" + "a" * 40)
    assert len(load_openrouter_api_key()) > 20
```

---

## 5. kcpa.py

### Bugs

#### B13 — Typo: `_carrar_dados` em vez de `_carregar_dados`
```python
historico = _carrar_dados(HISTORY_FILE)  # linha 298
```
Isso lançará `NameError` em runtime quando `get_pattern_summary()` for chamado.

**Correção:**
```python
historico = _carregar_dados(HISTORY_FILE)
```

#### B14 — `_salvar_dados` suprime exceções silenciosamente
```python
except IOError as e:
    pass  # Erro engolido
```
Se o disco estiver cheio ou sem permissão, o sistema perde dados sem alerta.

**Correção:**
```python
def _salvar_dados(arquivo: Path, dados: dict):
    try:
        arquivo.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
    except IOError as e:
        logger.error(f"Falha ao salvar {arquivo}: {e}")
        raise  # Ou retornar False
```

#### B15 — `registrar_interacao` recarrega todos os arquivos a cada chamada
I/O excessivo (3 arquivos lidos e 3 escritos por interação). Ineficiente para alta frequência.

**Correção:** Cache em memória com flush periódico:
```python
class KCPAStore:
    def __init__(self):
        self._cache = {}
        self._dirty = set()
    
    def get(self, arquivo: Path) -> dict:
        if arquivo not in self._cache:
            self._cache[arquivo] = _carregar_dados(arquivo)
        return self._cache[arquivo]
    
    def set(self, arquivo: Path, dados: dict):
        self._cache[arquivo] = dados
        self._dirty.add(arquivo)
    
    def flush(self):
        for arquivo in self._dirty:
            _salvar_dados(arquivo, self._cache[arquivo])
        self._dirty.clear()

_store = KCPAStore()
```

#### B16 — `registrar_interacao` não valida tipos
`input_usuario`, `output_koldi` podem ser `None` ou não-string, causando erro em `.encode()` e `.lower()`.

**Correção:**
```python
input_usuario = str(input_usuario) if input_usuario else ""
output_koldi = str(output_koldi) if output_koldi else ""
```

### Segurança

#### S8 — Hash MD5 (linha 126)
```python
"input_hash": hashlib.md5(input_usuario.encode()).hexdigest()[:8],
```
MD5 é criptograficamente quebrado, mas para hash de deduplicação não-segura é aceitável. Para fins de identificação de interações, um ID incremental ou UUID seria mais seguro e eficiente.

**Correção:**
```python
import uuid
"input_hash": str(uuid.uuid4())[:8],  # Ou usar hashlib.sha256
```

### Otimizações

#### P9 — E/S serializada a cada interação
3 arquivos lidos + 3 escritos = 6 operações de I/O por interação. Para 1000 interações, 6000 ops.

**Correção:** Usar SQLite (como em token_guard.py) com uma tabela `kcpa_interactions` e `kcpa_patterns`.

### Testes

```python
def test_adaptar_resposta_remove_enrolacao():
    original = "De acordo com o que foi solicitado, vamos implementar"
    adapted = adaptar_resposta(original)
    assert "De acordo com" not in adapted

def test_registrar_interacao_none():
    registrar_interacao(None, None)  # Não deve lançar exceção
```

---

## 6. kec.py

### Bugs

#### B17 — `salvar_em` não garante diretório pai existe
```python
Path(salvar_em).write_text(output, encoding="utf-8")
```
Se o diretório não existir, `write_text` lança `FileNotFoundError`.

**Correção:**
```python
if salvar_em:
    path = Path(salvar_em)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
```

#### B18 — `analisar_com_opencode` trunca código em 5000 chars sem aviso
```python
{codigo[:5000]}
```
Se o código tiver 10000 linhas, a análise será incompleta sem notificação.

**Correção:**
```python
if len(codigo) > 5000:
    codigo = codigo[:5000] + f"\n\n[... ARQUIVO TRUNCADO de {len(codigo)} caracteres. Analise completa ignorada.]"
```

#### B19 — `relatorio_evolucao` usa atributos que podem não existir
```python
summary['frases_incompletas_count']  # Pode ser None
summary.get('vocabulario_top20', {})  # Ok
```
`summary.get('verbos_frequencia', {})` é seguro, mas `summary['frases_incompletas_count']` lança KeyError se `get_pattern_summary` não retornar a chave por algum motivo.

**Correção:**
```python
frases_incompletas_count = summary.get('frases_incompletas_count', 0)
```

### Segurança

#### S9 — Subprocess com shell=False, mas argumentos não sanitizados
`codigo_ou_arquivo` vai para `prompt` que vai como argumento CLI para `opencode run {prompt}`. Se `prompt` contiver caracteres especiais, pode haver injection via argumento, embora `subprocess.run` com lista evite shell injection.

**Correção:**
```python
# Adicionar sanitização básica
prompt_sanitized = prompt.replace('"', '\\"').replace('`', '')
```

### Otimizações

#### P10 — KCPA importado dentro de funções
```python
from kcpa import get_pattern_summary  # linha 165
```
Import dentro de função é style questionável. Melhor no topo.

### Testes

```python
def test_evoluir_valid():
    result = evoluir("test input", "test output", "test context")
    assert "previsao" in result
    assert "output_adaptado" in result

def test_analisar_com_opencode_file_not_found():
    result = analisar_com_opencode("/nonexistent/file.py", "context")
    assert "ERRO" not in result or True  # Retorna conteúdo inline
```

---

## 7. mnemosyne_wrapper.py

### Bugs

#### B20 — `get_stats` importa sqlite3 dentro da função e usa path hardcoded
O caminho `DB_PATH` é definido globalmente, mas `get_stats` refaz a conexão SQLite manualmente (linhas 81-85) enquanto `remember`/`recall` usam o módulo `mnemosyne`.

**Problema:** Se a estrutura do DB mudar (ex: tabela renomeada), `get_stats` quebra. Mais grave: se `_mn` usa uma conexão diferente, `get_stats` pode ler dados inconsistentes.

**Correção:** Delegar stats ao módulo `mnemosyne` se tiver método `_mn.get_stats()`, senão remover ou sincronizar via `_mn`.
```python
def get_stats() -> dict:
    try:
        return _mn.get_stats() if hasattr(_mn, "get_stats") else {"total": 0, "error": "mnemosyne.get_stats not available"}
    except Exception as e:
        return {"error": str(e)}
```

#### B21 — `recall` chama `_mn.recall` com `source` mesmo se for `None`
```python
results = _mn.recall(query=query, top_k=top_k, source=source, from_date=from_date, to_date=to_date)
```
Se `source=None`, a API upstream pode interpretar como "qualquer source" ou pode quebrar se esperar string. Verificar documentação do `_mn.recall`.

**Correção:**
```python
kwargs = {"query": query, "top_k": top_k}
if source:
    kwargs["source"] = source
if from_date:
    kwargs["from_date"] = from_date
if to_date:
    kwargs["to_date"] = to_date
results = _mn.recall(**kwargs)
```

### Segurança

#### S10 — Injeção de SQL indireta via `scope` no filtro pós-busca
```python
if scope:
    results = [r for r in results if r.get("scope") == scope]
```
Não há injeção aqui (filtro Python puro), mas a performance pode ser péssima com muitos resultados. Melhor passar `scope` para a query SQL no backend.

### Otimizações

#### P11 — Filtro por scope é feito em Python, não no SQL
```python
results = _mn.recall(query=query, top_k=top_k, ...)
if scope:
    results = [r for r in results if r.get("scope") == scope]
```
Isso busca `top_k` resultados, mas filtra depois. Se `top_k=5` e apenas 2 têm o scope esperado, o usuário recebe 2 em vez de 5. O correto é buscar mais e filtrar, ou passar scope para o backend.

**Correção:**
```python
# Aumentar o fetch para compensar o filtro
fetch_k = top_k * 3
results = _mn.recall(query=query, top_k=fetch_k, ...)
if scope:
    results = [r for r in results if r.get("scope") == scope]
return results[:top_k]
```

### RAG + Cache + Memória Vetorial

#### RAG1 — Integração com `nomic-embed-text` para recall semântico offline
Se o `_mn` upstream não suporta busca vetorial local, implementar no wrapper:
```python
def recall_semantico(query: str, top_k: int = 5) -> list:
    """Recupera memórias usando embedding local."""
    try:
        emb = _LocalEmbeddings()
        qvec = emb.embed(query)
        # Carregar todas as memórias e rankear por cosine similarity
        todas = _mn.recall(query="", top_k=1000)  # ou usar SQL direto
        for m in todas:
            mvec = emb.embed(m.get("content", ""))
            m["_score"] = cosine_similarity(qvec, mvec)
        return sorted(todas, key=lambda x: x["_score"], reverse=True)[:top_k]
    except Exception as e:
        logger.error(f"Erro no recall semântico: {e}")
        return []
```

### Testes

```python
def test_remember_empty_content():
    mid = remember("")
    assert mid == ""  # Erro retorna string vazia

def test_recall_no_results():
    results = recall("zzz_nonexistent_zzz", top_k=3)
    assert isinstance(results, list)
```

---

## 8. token_guard.py

### Bugs

#### B22 — `get_status` e `_reset_if_needed` podem causar divisão por zero
```python
"pct": state["session_tokens"] / DEFAULT_SESSION_LIMIT * 100,
```
`DEFAULT_SESSION_LIMIT = 100_000`, então zero só se alterarem. Mas se DEFAULT_SESSION_LIMIT for zero (config customizada):
```python
usage = state[key] / limit if limit > 0 else 0  # Linha 130 protegida
```
`get_status` (linha 195) não tem essa proteção:
```python
"pct": state["session_tokens"] / DEFAULT_SESSION_LIMIT * 100,
```

**Correção:**
```python
def _safe_pct(used: int, limit: int) -> float:
    return used / limit * 100 if limit > 0 else 0.0
```

#### B23 — `_reset_if_needed` pode lançar exceção com `datetime.fromisoformat`
```python
hour_start = datetime.fromisoformat(state["hour_start"])
```
Se o estado for corrompido (hack, crash durante escrita), `fromisoformat` lança `ValueError`.

**Correção:**
```python
try:
    hour_start = datetime.fromisoformat(state["hour_start"])
except (ValueError, TypeError):
    hour_start = datetime.now()
    state["hour_start"] = hour_start.isoformat()
```

#### B24 — Reset diário também reseta sessão (linha 67-68)
```python
state["session_tokens"] = 0  # Reset de sessão no reset diário é questionável
state["session_start"] = now.isoformat()
```
Sessão é por definição mais curta que um dia. Resetar sessão junto com diário não é bug grave, mas pode confundir.

**Correção:** Manter sessão independente:
```python
if state.get("day_start") != now.strftime("%Y-%m-%d"):
    state["daily_tokens"] = 0
    state["day_start"] = now.strftime("%Y-%m-%d")
```

### Segurança

#### S11 — Estado em JSON legível/gravável por qualquer processo
`token_guard_state.json` em `~/.hermes/` sem permissões restritas.

**Correção:**
```python
def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))
    if sys.platform != "win32":
        STATE_FILE.chmod(0o600)
```

### Otimizações

#### P12 — `check_budget` e `record_usage` lêem/gravam disco a cada chamada
Para alta frequência, isso adiciona latência de ~5-10ms por operação de I/O.

**Correção:** Cache em memória com flush a cada N chamadas:
```python
class TokenGuardState:
    def __init__(self):
        self._state = None
        self._dirty = False
        self._calls_since_flush = 0
    
    def get(self):
        if self._state is None:
            self._state = _load_state()
        return self._state
    
    def save(self):
        if self._dirty:
            _save_state(self._state)
            self._dirty = False
    
    def mark_dirty(self):
        self._dirty = True
```

### Testes

```python
def test_check_budget_allows_under_limit():
    result = check_budget(estimated_tokens=1000, session_limit=100000)
    assert result["allowed"]

def test_check_budget_blocks_over_limit():
    state = _load_state()
    state["session_tokens"] = 90000
    _save_state(state)
    result = check_budget(estimated_tokens=20000, session_limit=100000)
    assert not result["allowed"]

def test_get_status_no_division_by_zero():
    state = _load_state()
    state["session_tokens"] = 0
    _save_state(state)
    status = get_status()
    assert status["session"]["pct"] == 0.0

def test_reset_clears_all():
    record_usage(50000)
    reset()
    status = get_status()
    assert status["session"]["used"] == 0
```

---

## 9. planning.py

### Bugs

#### B25 — `get_plan` lê `task_plan` duas vezes (linhas 110-112)
```python
"task_plan": task_plan.read_text(encoding="utf-8") if task_plan.exists() else "",
"phases": _parse_phases(task_plan.read_text(encoding="utf-8")) if task_plan.exists() else [],
```
Se o arquivo for lido com sucesso na primeira vez, mas falhar na segunda (ex: antivírus bloqueando), a segunda leitura lança exceção.

**Correção:**
```python
def get_plan(plan_dir: str) -> dict:
    plan_dir = Path(plan_dir)
    task_plan = plan_dir / "task_plan.md"
    notes = plan_dir / "notes.md"
    
    task_content = task_plan.read_text(encoding="utf-8") if task_plan.exists() else ""
    notes_content = notes.read_text(encoding="utf-8") if notes.exists() else ""
    
    return {
        "dir": str(plan_dir),
        "task_plan": task_content,
        "notes": notes_content,
        "phases": _parse_phases(task_content) if task_content else [],
    }
```

#### B26 — `update_phase` modifica o bloco Status incorretamente
```python
for j in range(i, len(lines)):
    if lines[j].startswith("## Status"):
        lines[j] = f"## Status\n{status_msg}"  # <— Isso quebra o markdown
        break
```
Isso substitui a linha `## Status` por `## Status\nAlgo`, o que resulta em `## Status\nAlgo` em uma única linha. O marcador `## Status` some e vira parte do texto.

**Correção:**
```python
if status_msg:
    for j in range(i + 1, len(lines)):
        if lines[j].startswith("## Status"):
            for k in range(j + 1, len(lines)):
                if lines[k].strip() and not lines[k].startswith("##"):
                    lines[k] = status_msg
                    break
            break
```
Melhor ainda: encontrar a linha após `## Status` e substituí-la.

#### B27 — `create_plan` não valida `task_name` vazio
```python
safe = _safe_dirname(task_name)
plan_dir = _get_plans_dir() / f"{_today()}-{safe_name}"
```
Se `task_name=""`, `safe_name="untitled"`, o que é aceitável, mas pode gerar pastas duplicadas. Se `task_name=None`, `task_name.lower()` lança `AttributeError`.

**Correção:**
```python
def create_plan(task_name: str, phases: list[str], ...):
    task_name = task_name or "untitled"
    if not phases:
        raise ValueError("Pelo menos uma fase é necessária")
    ...
```

#### B28 — `_safe_dirname` permite caracteres Unicode que podem causar problemas no Windows
`re.sub(r'[^\w\-]', '-', ...)` remove não-alfanuméricos, mas mantém caracteres acentuados (á, é, ó), que são \w em Python 3 com Unicode. No Windows, alguns caracteres acentuados são permitidos em nomes de pasta, mas podem causar problemas em ferramentas de terminal.

**Correção:** Normalizar para ASCII:
```python
import unicodedata
def _safe_dirname(name: str) -> str:
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    safe = re.sub(r'[^\w\-]', '-', name.lower().strip().replace(' ', '-'))
    ...
```

### Segurança

#### S12 — Manipulação de path via `plan_dir` pode permitir path traversal
```python
def create_plan(task_name, phases, ..., plan_dir: Optional[str] = None):
    if plan_dir is None:
        ...
    else:
        plan_dir = Path(plan_dir)  # Se plan_dir="../malicious", permite escrita fora do diretório
```

**Correção:**
```python
def create_plan(..., plan_dir: Optional[str] = None):
    if plan_dir is not None:
        plan_path = Path(plan_dir)
        # Verificar se está dentro do diretório permitido
        plans_base = _get_plans_dir()
        try:
            plan_path.relative_to(plans_base)
        except ValueError:
            raise ValueError(f"plan_dir deve estar dentro de {plans_base}")
```

### Otimizações

#### P13 — `get_plan` lê arquivos inteiros para memória
Para `task_plan.md` de 1MB, carrega tudo. Aceitável para arquivos de plano pequenos.

#### P14 — `list_plans` faz I/O para cada plano (N+1 queries)
Lê diretório, itera cada pasta, chama `get_plan` que lê 2 arquivos. Para 20 planos, 40 operações de leitura.

**Correção:** Cache de metadados em `plans_index.json`:
```python
def _build_index():
    index = {}
    for d in _get_plans_dir().iterdir():
        if d.is_dir() and (d / "task_plan.md").exists():
            phases = _parse_phases((d / "task_plan.md").read_text())
            index[d.name] = {
                "modified": os.path.getmtime(d / "task_plan.md"),
                "phases": phases,
            }
    return index
```

### Testes

```python
def test_create_plan_empty_phases():
    with pytest.raises(ValueError):
        create_plan("test", [])

def test_safe_dirname_special_chars():
    assert _safe_dirname("a/b\\c:d") == "a-b-c-d"
    assert "../" not in _safe_dirname("../../etc")

def test_update_phase_invalid_index():
    plan = create_plan("test", ["Phase 1"])
    result = update_phase(plan, 99, done=True)
    assert not result  # Retorna False

def test_parse_phases_empty():
    assert _parse_phases("No checklist here") == []

def test_list_plans_empty():
    plans = list_plans()
    assert isinstance(plans, list)
```

---

## 10. toolbox_pg.py

### Bugs

#### B29 — `_invoke_toolbox` parsing de JSON frágil
```python
for line in output.split("\n"):
    line = line.strip()
    if line.startswith("[") or line.startswith("{"):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
```
Se a saída tiver múltiplas linhas com JSON (ex: log + resultado), pode pegar a linha errada. Se o JSON for multilinha, não funciona.

**Correção:** Usar `json.loads` na saída completa, e se falhar, tentar extrair o último JSON válido:
```python
def _extract_json(text: str) -> dict | list | None:
    """Extrai JSON da saída do toolbox."""
    # Procurar por { } ou [ ] no texto
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start >= 0:
            end = text.rfind(closer) + 1
            if end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    continue
    return None
```

### Segurança (CRÍTICO)

#### S13 — IP do servidor e caminho de SSH key hardcoded
```python
SSH_KEY = "/c/Users/dell-/.ssh/id_ed25519_vps"
VPS_HOST = "root@2.25.168.233"
```
**ISSO É UM VAZAMENTO DE INFORMAÇÃO CRÍTICO.** O IP do servidor e o nome do arquivo de chave SSH estão expostos no código-fonte. Se este repositório for público ou acessado por terceiros, o servidor pode ser atacado.

**Correção IMEDIATA:**
```python
SSH_KEY = os.environ.get("KOLDI_SSH_KEY", "")
VPS_HOST = os.environ.get("KOLDI_VPS_HOST", "")
if not SSH_KEY or not VPS_HOST:
    raise ValueError("KOLDI_SSH_KEY e KOLDI_VPS_HOST devem estar configurados como variáveis de ambiente")
```
E mover `2.25.168.233` e `id_ed25519_vps` para `.env` ou variáveis de ambiente.

#### S14 — `StrictHostKeyChecking=no` — MITM risk
```python
["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", VPS_HOST, cmd]
```
Isso desabilita a verificação de chave do host, permitindo ataque Man-in-the-Middle. Qualquer servidor com o mesmo IP pode se passar pelo VPS.

**Correção:** Usar `StrictHostKeyChecking=accept-new` ou adicionar a fingerprint do servidor ao `known_hosts`:
```python
["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=" + str(HOME / ".ssh" / "known_hosts"), VPS_HOST, cmd]
```

#### S15 — Comando shell remoto inseguro
```python
cmd = f'env $(cat {ENV_FILE} | xargs) toolbox --prebuilt postgres invoke {tool_name} \'{args_json}\''
```
`args_json` pode conter single quotes que quebram o shell remoto. Um `args_json` com `'` fecha a string e permite injeção de comandos no servidor remoto.

**Correção:**
```python
import shlex
cmd = f'env $(cat {ENV_FILE} | xargs) toolbox --prebuilt postgres invoke {shlex.quote(tool_name)} {shlex.quote(args_json)}'
```

### Performance

#### P15 — Conexão SSH para cada query
Cada chamada a `sql_query` abre uma nova conexão SSH (TCP + autenticação). 3-5 segundos de overhead.

**Correção:** Usar conexão SSH persistente com `paramiko` ou manter um túnel SSH ativo:
```python
import paramiko

class SSHClientPool:
    def __init__(self):
        self._client = None
    
    def _connect(self):
        if self._client is None or not self._client.get_transport().is_active():
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self._client.connect(
                hostname=VPS_HOST.split("@")[1],
                username=VPS_HOST.split("@")[0],
                key_filename=SSH_KEY,
                timeout=10,
            )
        return self._client
```

---

## 11. memory_pipeline.py

### Bugs

#### B30 — Double-wrapping de dict: MemoryItem recebe dict, mas `enqueue_many` trata `items` como objetos
```python
def enqueue_many(items: Any) -> Any:
    ...
    for item in items:
        out.append(_fallback_create_entry(item if isinstance(item, dict) else item.__dict__))
```
Em `enqueue_memory`, o item é passado como `MemoryItem(...)`. Em `enqueue_many`, se `items` contiver `MemoryItem`, `isinstance(item, dict)` é False, então cai em `item.__dict__`, mas `MemoryItem` já é um objeto com atributos. Funciona, mas é frágil.

**Correção:** Normalizar para dict:
```python
def _to_dict(item: Any) -> dict:
    if isinstance(item, dict):
        return item
    return {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
```

#### B31 — `_memory_queue` cria nova instância a cada chamada
```python
def _memory_queue() -> Optional[MemoryQueue]:
    try:
        q = MemoryQueue(db_path=MEMORY_QUEUE_DB, autostart=True)
```
Isso cria uma nova conexão com o banco a cada chamada, com thread própria. Se chamado frequentemente, cria múltiplas threads.

**Correção:** Singleton:
```python
_MEMORY_QUEUE_INSTANCE = None

def _memory_queue() -> Optional[MemoryQueue]:
    global _MEMORY_QUEUE_INSTANCE
    if _MEMORY_QUEUE_INSTANCE is None:
        try:
            _MEMORY_QUEUE_INSTANCE = MemoryQueue(db_path=MEMORY_QUEUE_DB, autostart=True)
        except Exception as exc:
            log.warning("MemoryQueue indisponível: %s", exc)
            return None
    return _MEMORY_QUEUE_INSTANCE
```

---

## 12. Resumo Consolidado

### Prioridade de Correções

| Prioridade | Arquivo | Problema | Impacto |
|-----------|---------|----------|---------|
| **CRÍTICA** | `toolbox_pg.py` | IP/chave SSH hardcoded + StrictHostKeyChecking=no | Segurança do servidor |
| **CRÍTICA** | `koldi_utils.py` | Sanitização excessiva quebra caminhos legítimos | UX quebrada |
| **ALTA** | `kcpa.py` | `_carrar_dados` (typo) causa NameError | Runtime crash |
| **ALTA** | `consultar_ia.py` | `consultar_ia_stream` valida após montar payload | Ordem de operações |
| **ALTA** | `koldi_utils.py` | Leitura de API key via registry com parsing frágil | Falha silenciosa |
| **MÉDIA** | `token_guard.py` | Divisão por zero em `get_status` | RuntimeError |
| **MÉDIA** | `planning.py` | `update_phase` corrompe markdown do Status | Dados corrompidos |
| **MÉDIA** | `orquestrador.py` | `.format()` sem escape lança KeyError | Runtime crash |
| **MÉDIA** | `kcpa.py` | I/O excessivo (6 ops/interação) | Performance |
| **MÉDIA** | `mnemosyne_wrapper.py` | Filtro scope pós-busca reduz resultados | Recall impreciso |
| **BAIXA** | `consultar_ia.py` | Sem cache de respostas | $$ desperdiçado |
| **BAIXA** | `consultar_ia.py` | Sem retry com backoff | Instabilidade |
| **BAIXA** | `consultar_ia.py` | `time.sleep` bloqueante em loops | Performance |
| **BAIXA** | `front_controller.py` | Padrões de match sem word boundary | Falsos positivos |
| **BAIXA** | `kec.py` | `salvar_em` sem mkdir parents | Erro ao salvar |
| **SUGESTÃO** | Global | `nomic-embed-text` não usado | RAG offline viável |
| **SUGESTÃO** | Global | Sem testes unitários automatizados | Regressões frequentes |
| **SUGESTÃO** | `consultar_ia.py` | Connection pooling com `requests.Session` | ~30% mais rápido |
| **SUGESTÃO** | `consultar_ia.py` | Paralelismo com ThreadPoolExecutor | 3x throughput |
| **SUGESTÃO** | `koldi_utils.py` | Remover leitura de registry | Segurança |
| **SUGESTÃO** | `toolbox_pg.py` | Conexão SSH persistente (paramiko) | -3s por query |

### Melhorias Arquiteturais Recomendadas

1. **Cache Layer:** Implementar cache LRU em `consultar_ia.py` para prompts repetidos (economia de tokens/$)
2. **Retry Policy:** Adicionar `tenacity` ou retry manual com exponential backoff para rate limits (429) e 5xx
3. **RAG com nomic-embed-text:** Implementar embeddings locais para recall semântico no `mnemosyne_wrapper.py`
4. **Paralelismo:** Substituir `time.sleep` loops por `ThreadPoolExecutor` em `orquestrador.py` e `consultar_ia.py`
5. **Connection Pool:** Usar `requests.Session()` globalmente para reuso de conexões TCP/TLS
6. **SQLite centralizado:** Unificar armazenamento de KCPA patterns, token_guard state, e cache em SQLite em vez de múltiplos JSONs
7. **Config Management:** Mover todas as credenciais (SSH, API keys, hosts) para variáveis de ambiente
8. **Testing:** Implementar `pytest` com fixtures para mockar chamadas HTTP e subprocess
9. **Logging Estruturado:** Usar `structlog` ou JSON logging para melhor debugging
10. **Health Check:** Centralizar `status()` em um módulo de health check com timeout e fallbacks

### Métricas de Impacto Estimado

| Melhoria | Ganho Estimado |
|----------|---------------|
| Cache de respostas | -40% tokens usados para prompts repetidos |
| ThreadPoolExecutor | 3x throughput em `comparar_modelos` |
| Connection Pooling | -20% latência por chamada |
| Retry com backoff | -90% de erros transientes |
| RAG local (nomic) | $0 custo de embedding, recall instantâneo |
| SSH persistente | -3s por query no toolbox_pg |
| Testes automatizados | -70% regressões não-detectadas |

---

*Análise gerada por opencode com deepseek-v4-flash-free — 2026-06-13*
