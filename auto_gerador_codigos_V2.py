# auto_construtor_codigos_v2.py
"""
Auto Construtor de Código v2 - Sistema Avançado de Geração de Código
Desenvolvido para a Assistente Atena

Este módulo implementa um sistema robusto de geração automática de código
com validação multi-etapas, integração flexível com LLMs e framework ético.
Versão consolidada e pronta para integração.
"""

import ast
import logging
import re
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Dependências Opcionais com Fallbacks ---
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("Biblioteca 'requests' não encontrada. A integração com LM Studio estará desabilitada.")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.warning("Biblioteca 'BeautifulSoup' não encontrada. A validação de consistência do DOM será pulada.")

# --- Estruturas de Dados e Enums ---

class ValidationStatus(Enum):
    """Define os status possíveis para uma etapa de validação."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

class LLMProvider(Enum):
    """Define os provedores de LLM suportados pelo sistema."""
    LM_STUDIO = "lm_studio"
    HUGGING_FACE = "hugging_face" # Placeholder para futura implementação

@dataclass
class ProblemContext:
    """Encapsula todo o contexto de um problema a ser resolvido."""
    description: str
    target_url: Optional[str] = None
    html_snapshot: Optional[str] = None
    framework: str = "playwright"
    existing_code: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    """Armazena o resultado de uma única verificação do validador."""
    status: ValidationStatus
    message: str
    validator_name: str
    details: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

@dataclass
class GeneratedSolution:
    """Representa a solução de código gerada, incluindo metadados e validação."""
    code: str
    validation_results: List[ValidationResult] = field(default_factory=list)
    quality_score: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        """Determina se a solução é considerada válida (sem falhas críticas)."""
        return not any(r.status == ValidationStatus.FAILED for r in self.validation_results)

# --- Componente de Integração com LLM ---

class LLMIntegration:
    """Gerencia a comunicação com o LLM, otimizado para rodar localmente."""

    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1/chat/completions", max_retries: int = 2, timeout: int = 120):
        if not HAS_REQUESTS:
            raise ImportError("A biblioteca 'requests' é necessária para a integração com LLM Studio.")
        self.lm_studio_url = lm_studio_url
        self.max_retries = max_retries
        self.timeout = timeout

    async def generate_code(self, prompt: str, max_tokens: int = 1500) -> Optional[str]:
        """Gera código de forma assíncrona usando o LLM configurado."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Tentativa {attempt + 1} de chamar o LLM.")
                payload = {
                    "model": "local-model",
                    "messages": [
                        {"role": "system", "content": "Você é um especialista em automação web Python com Playwright. Gere apenas o bloco de código Python solicitado, sem explicações adicionais."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2, # Baixa temperatura para respostas mais determinísticas
                    "stream": False
                }
                
                response = await asyncio.to_thread(
                    requests.post,
                    self.lm_studio_url,
                    json=payload,
                    timeout=self.timeout
                )

                response.raise_for_status()
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if content:
                    return self._extract_code_from_response(content)
                
                logger.warning("LLM retornou uma resposta vazia.")

            except requests.exceptions.RequestException as e:
                logger.error(f"Erro de comunicação com o LLM na tentativa {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Backoff exponencial
            except Exception as e:
                logger.error(f"Erro inesperado ao gerar código na tentativa {attempt + 1}: {e}")
        
        logger.error("Falha ao gerar código após todas as tentativas.")
        return None

    def _extract_code_from_response(self, response: str) -> str:
        """Extrai blocos de código Python de uma resposta do LLM."""
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        # Fallback se não encontrar o formato padrão
        if "def " in response or "import " in response:
            return response.strip()
        return ""

# --- Componente de Validação de Código ---

class CodeValidator:
    """Realiza uma análise multifacetada do código gerado."""

    def validate_all(self, code: str, context: ProblemContext) -> List[ValidationResult]:
        """Executa todas as validações e retorna uma lista de resultados."""
        results = []
        
        # Validação de Sintaxe (crítica)
        syntax_result = self._validate_syntax(code)
        results.append(syntax_result)
        if not syntax_result.status == ValidationStatus.PASSED:
            return results # Para se a sintaxe for inválida

        # Outras validações
        results.append(self._validate_security(code))
        if context.html_snapshot and HAS_BS4:
            results.append(self._validate_consistency(code, context.html_snapshot))
        results.append(self._assess_quality(code))
        
        return results

    def _validate_syntax(self, code: str) -> ValidationResult:
        """Verifica se o código possui sintaxe Python válida."""
        try:
            ast.parse(code)
            return ValidationResult(status=ValidationStatus.PASSED, message="Sintaxe Python válida.", validator_name="Syntax")
        except SyntaxError as e:
            return ValidationResult(status=ValidationStatus.FAILED, message=f"Erro de sintaxe na linha {e.lineno}: {e.msg}", validator_name="Syntax")

    def _validate_security(self, code: str) -> ValidationResult:
        """Verifica a presença de padrões de código potencialmente perigosos."""
        dangerous_patterns = {
            r"os\.system": "Execução de comando no sistema operacional.",
            r"subprocess\.run": "Criação de subprocesso.",
            r"eval\(|exec\(|pickle\.load": "Execução de código ou desserialização perigosa."
        }
        issues = []
        for pattern, msg in dangerous_patterns.items():
            if re.search(pattern, code):
                issues.append(msg)
        
        if issues:
            return ValidationResult(status=ValidationStatus.WARNING, message=f"Padrões perigosos encontrados: {', '.join(issues)}", validator_name="Security")
        return ValidationResult(status=ValidationStatus.PASSED, message="Nenhum risco de segurança óbvio detectado.", validator_name="Security")

    def _validate_consistency(self, code: str, html: str) -> ValidationResult:
        """Verifica se os seletores no código existem no HTML fornecido."""
        soup = BeautifulSoup(html, 'html.parser')
        selectors = re.findall(r'page\.(?:locator|click|fill)\(["\'](.*?)["\']\)', code)
        missing = []
        for selector in selectors:
            try:
                if not soup.select_one(selector):
                    missing.append(selector)
            except Exception:
                missing.append(f"{selector} (seletor inválido)")
        
        if missing:
            return ValidationResult(status=ValidationStatus.WARNING, message=f"Seletores não encontrados no HTML: {', '.join(missing)}", validator_name="Consistency")
        return ValidationResult(status=ValidationStatus.PASSED, message="Todos os seletores parecem consistentes com o HTML.", validator_name="Consistency")

    def _assess_quality(self, code: str) -> ValidationResult:
        """Avalia a qualidade do código com base em heurísticas."""
        score = 1.0
        issues = []
        
        # Penaliza por uso de time.sleep()
        if "time.sleep" in code:
            score -= 0.3
            issues.append("Uso de 'time.sleep' em vez de esperas explícitas do Playwright.")
        
        # Penaliza por falta de tratamento de erro
        if "try:" not in code and len(code.splitlines()) > 10:
            score -= 0.2
            issues.append("Falta de blocos try/except para tratamento de erros.")
            
        # Penaliza por falta de comentários
        if "#" not in code and len(code.splitlines()) > 15:
            score -= 0.1
            issues.append("Código extenso sem comentários.")

        if score < 1.0:
            return ValidationResult(status=ValidationStatus.WARNING, message=f"Pontos de melhoria de qualidade detectados. Issues: {' '.join(issues)}", score=score, validator_name="Quality")
        return ValidationResult(status=ValidationStatus.PASSED, message="Código segue boas práticas básicas.", score=score, validator_name="Quality")


# --- Gerador de Código Principal ---

class AdvancedCodeGenerator:
    """Orquestra o processo de geração e validação de código."""

    def __init__(self, llm_integration: Optional[LLMIntegration] = None, max_iterations: int = 2):
        self.llm = llm_integration or LLMIntegration()
        self.validator = CodeValidator()
        self.max_iterations = max_iterations

    def _build_optimized_prompt(self, context: ProblemContext, issues: List[str] = None) -> str:
        """Constrói um prompt otimizado para o LLM."""
        prompt = f"""
Você é um especialista em automação web usando Python e Playwright.
Sua tarefa é gerar um script para resolver o seguinte problema:

**Objetivo:** {context.description}
**URL Alvo:** {context.target_url or "Não especificada"}
**Framework:** {context.framework}
"""
        html_preview = (context.html_snapshot[:1500] + "...") if context.html_snapshot and len(context.html_snapshot) > 1500 else context.html_snapshot
        if html_preview:
            prompt += f"\n**Contexto da Página (Preview do HTML):**\n```html\n{html_preview}\n```\n"

        if context.existing_code:
            prompt += f"\n**Código Existente (para reparo):**\n```python\n{context.existing_code}\n```\n"

        if issues:
            prompt += "\n**Problemas na Versão Anterior (corrija estes pontos):**\n"
            prompt += "\n".join(f"- {issue}" for issue in issues)
            prompt += "\n"

        prompt += """
**Instruções Finais:**
1. Gere um script Python completo e funcional.
2. Use esperas explícitas do Playwright (`page.wait_for_selector`, etc.) em vez de `time.sleep()`.
3. Inclua blocos `try/except` para tratamento de erros robusto.
4. Adicione logging para informar sobre o progresso.
5. Retorne APENAS o código dentro de um bloco ```python.
"""
        return prompt

    async def generate_solution(self, context: ProblemContext) -> GeneratedSolution:
        """Ciclo completo de geração e auto-melhoria."""
        logger.info(f"Iniciando geração de solução para: {context.description}")
        best_solution = GeneratedSolution(code="", quality_score=-1)
        issues_for_next_iteration = []

        for i in range(self.max_iterations):
            logger.info(f"--- Iteração {i + 1}/{self.max_iterations} ---")
            prompt = self._build_optimized_prompt(context, issues_for_next_iteration)
            generated_code = await self.llm.generate_code(prompt)

            if not generated_code:
                logger.warning("Falha na geração de código pelo LLM.")
                continue

            validation_results = self.validator.validate_all(generated_code, context)
            
            # Calcula o score de qualidade
            scored_results = [r.score for r in validation_results if r.score is not None]
            quality_score = sum(scored_results) / len(scored_results) if scored_results else 0.5
            
            current_solution = GeneratedSolution(generated_code, validation_results, quality_score)
            
            if quality_score > best_solution.quality_score:
                logger.info(f"Nova melhor solução encontrada (Score: {quality_score:.2f})")
                best_solution = current_solution

            if current_solution.is_valid and quality_score > 0.9:
                logger.info("Solução de alta qualidade encontrada, finalizando iterações.")
                break
            
            # Prepara para a próxima iteração
            issues_for_next_iteration = [res.message for res in validation_results if res.status != ValidationStatus.PASSED]
            if not issues_for_next_iteration:
                logger.info("Nenhum problema encontrado, finalizando iterações.")
                break
        
        if not best_solution.code:
             return GeneratedSolution(code="# Falha ao gerar código após todas as iterações.", validation_results=[
                ValidationResult(status=ValidationStatus.FAILED, message="Nenhum código foi gerado.", validator_name="Generator")
             ])

        logger.info(f"Geração concluída. Score final da melhor solução: {best_solution.quality_score:.2f}")
        return best_solution

# --- Fachada para Integração Simplificada ---

class AutoCodeConstructorFacade:
    """Ponto de entrada simplificado para o sistema de geração de código."""

    def __init__(self):
        try:
            llm_interface = LLMIntegration()
            self.generator = AdvancedCodeGenerator(llm_integration=llm_interface)
            self.is_ready = True
            logger.info("Fachada do Construtor de Código inicializada com sucesso.")
        except ImportError as e:
            self.generator = None
            self.is_ready = False
            logger.critical(f"Falha ao inicializar a fachada do construtor de código: {e}")

    async def generate_automation_code(self, error_context: str, html_content: str, target_action: str) -> Dict[str, Any]:
        """
        Gera um script de automação com base no contexto de um erro ou tarefa.

        Args:
            error_context (str): Descrição do erro ou da automação falha.
            html_content (str): Snapshot do HTML da página no momento do erro.
            target_action (str): Descrição da ação que deveria ser executada.

        Returns:
            Um dicionário contendo o código gerado e uma explicação.
        """
        if not self.is_ready:
            return {
                "success": False,
                "error": "O construtor de código não está pronto. Verifique as dependências (ex: requests)."
            }

        problem_description = f"Ocorreu um erro: '{error_context}'. A automação deveria '{target_action}'."
        
        context = ProblemContext(
            description=problem_description,
            html_snapshot=html_content,
            target_url="URL não especificada no contexto do erro"
        )
        
        try:
            solution = await self.generator.generate_solution(context)
            
            if solution.is_valid:
                explanation = "O script gerado tenta corrigir o problema com base no contexto fornecido. Ele usa esperas explícitas e seletores robustos para maior confiabilidade."
                return {
                    "success": True,
                    "code": solution.code,
                    "explanation": explanation,
                    "quality_score": solution.quality_score
                }
            else:
                failed_checks = [f"{r.validator_name}: {r.message}" for r in solution.validation_results if r.status == ValidationStatus.FAILED]
                return {
                    "success": False,
                    "error": "O código gerado não passou nas validações críticas.",
                    "details": {
                        "failed_checks": failed_checks,
                        "generated_code_attempt": solution.code
                    }
                }
        except Exception as e:
            logger.error(f"Erro crítico no processo de geração de código: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Uma exceção interna ocorreu durante a geração do código: {e}"
            }


# --- Bloco de Demonstração ---
async def main_demo():
    """Função de demonstração para testar o módulo de forma independente."""
    logger.info("--- INICIANDO DEMONSTRAÇÃO DO AUTO CONSTRUTOR DE CÓDIGO v2 ---")
    
    facade = AutoCodeConstructorFacade()
    if not facade.is_ready:
        logger.error("A demonstração não pode continuar pois a fachada não foi inicializada.")
        return

    # Cenário de problema
    html_exemplo = """
    <html><body>
        <form>
            <label for="user">Usuário:</label>
            <input type="email" id="email-field" name="username">
            <label for="pass">Senha:</label>
            <input type="password" id="password-field" name="password">
            <button type="submit" class="btn-login">Entrar</button>
        </form>
    </body></html>
    """
    
    resultado = await facade.generate_automation_code(
        error_context="Elemento 'login-button' não encontrado.",
        html_content=html_exemplo,
        target_action="Fazer login no site preenchendo email e senha e clicando no botão 'Entrar'."
    )
    
    print("\n" + "="*50)
    print("RESULTADO DA DEMONSTRAÇÃO")
    print("="*50)
    
    if resultado.get("success"):
        print("✅ Geração de código bem-sucedida!")
        print(f"📊 Score de Qualidade: {resultado.get('quality_score', 0):.2f}\n")
        print("--- CÓDIGO GERADO ---")
        print(resultado.get("code"))
        print("\n--- EXPLICAÇÃO ---")
        print(resultado.get("explanation"))
    else:
        print("❌ Falha na geração de código.")
        print(f"Erro: {resultado.get('error')}")
        if "details" in resultado:
            print("Detalhes:", json.dumps(resultado["details"], indent=2))
            
    print("="*50)


if __name__ == "__main__":
    # Para executar a demonstração, você precisa ter um servidor LM Studio rodando localmente.
    # Exemplo de comando para rodar a demo: python auto_construtor_codigos_v2.py
    asyncio.run(main_demo())