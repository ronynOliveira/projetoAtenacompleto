# atena_gerador_codigo.py
# Versão 1.0 - Módulo Unificado de Geração de Código

import ast
import logging
import re
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

# Dependências opcionais com fallbacks
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logging.warning("Biblioteca 'requests' não encontrada. A integração com LM Studio estará desabilitada.")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logging.warning("Biblioteca 'BeautifulSoup' não encontrada. A validação de consistência do DOM será pulada.")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    logging.warning("Biblioteca 'httpx' não encontrada. A integração assíncrona com LM Studio pode ser limitada.")

try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("Bibliotecas 'torch' ou 'transformers' não encontradas. A integração com Hugging Face estará desabilitada.")

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    HUGGING_FACE = "hugging_face"
    AUTO = "auto"

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
        self.lm_studio_url = lm_studio_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.hf_pipeline = None
        self.model_cache = {}

    def _initialize_huggingface_model(self, model_name: str = "microsoft/phi-1_5"):
        """Inicializa modelo Hugging Face para CPU."""
        if not HAS_TRANSFORMERS:
            logger.warning("Hugging Face não disponível. Não é possível inicializar o modelo.")
            return False
        try:
            if model_name not in self.model_cache:
                logger.info(f"Carregando modelo Hugging Face: {model_name}")
                
                device = "cpu"
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32, # Força float32 para CPU
                    device_map="cpu"
                )
                
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU
                    max_length=512,
                    temperature=0.7,
                    do_sample=True
                )
                
                self.model_cache[model_name] = self.hf_pipeline
                logger.info(f"Modelo {model_name} carregado com sucesso")
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo Hugging Face: {e}")
            return False

    async def _call_lm_studio(self, prompt: str) -> Dict[str, Any]:
        """Chama API do LM Studio."""
        if not HAS_HTTPX or not HAS_REQUESTS:
            return {'success': False, 'error': 'LM Studio não disponível (requests/httpx não instalados).'}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.lm_studio_url,
                    json={
                        "model": "local-model",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return {
                        'success': True,
                        'response': content,
                        'model': 'lm_studio'
                    }
                else:
                    return {
                        'success': False,
                        'error': f"LM Studio retornou status {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            logger.error(f"Erro na chamada ao LM Studio: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _call_huggingface(self, prompt: str) -> Dict[str, Any]:
        """Chama modelo Hugging Face."""
        if not HAS_TRANSFORMERS:
            return {'success': False, 'error': 'Hugging Face não disponível.'}

        try:
            if not self.hf_pipeline:
                if not self._initialize_huggingface_model():
                    return {
                        'success': False,
                        'error': 'Falha ao inicializar modelo Hugging Face'
                    }
            
            outputs = self.hf_pipeline(
                prompt,
                max_new_tokens=400, # Ajustado para ser mais consistente
                num_return_sequences=1,
                pad_token_id=self.hf_pipeline.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            if generated_text.startswith(prompt):
                 content = generated_text[len(prompt):].strip()
            else:
                 content = generated_text
            
            return {
                'success': True,
                'response': content,
                'model': 'huggingface'
            }
            
        except Exception as e:
            logger.error(f"Erro na chamada ao Hugging Face: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_code_from_response(self, response: str) -> str:
        """Extrai blocos de código Python de uma resposta do LLM."""
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        # Fallback se não encontrar o formato padrão
        if "def " in response or "import " in response:
            return response.strip()
        return ""

    async def generate_text_with_code(self, prompt: str, ai_model_preference: str = "auto") -> Dict[str, Any]:
        """Gera código usando o modelo especificado."""
        result = {
            'raw_response': '',
            'generated_code': '',
            'explanation': '',
            'used_model': '',
            'confidence_score': 0.0
        }
        
        # Tenta LM Studio primeiro se disponível
        if ai_model_preference in ["auto", "lm_studio"] and HAS_HTTPX and HAS_REQUESTS:
            lm_result = await self._call_lm_studio(prompt)
            if lm_result['success']:
                result['raw_response'] = lm_result['response']
                result['used_model'] = 'lm_studio'
                result['confidence_score'] = 0.9
                
                code_data = self._extract_code_from_response(lm_result['response'])
                result['generated_code'] = code_data
                result['explanation'] = "Código gerado via LM Studio."
                
                return result
        
        # Fallback para Hugging Face
        if ai_model_preference in ["auto", "huggingface"] and HAS_TRANSFORMERS:
            hf_result = self._call_huggingface(prompt)
            if hf_result['success']:
                result['raw_response'] = hf_result['response']
                result['used_model'] = 'huggingface'
                result['confidence_score'] = 0.7
                
                code_data = self._extract_code_from_response(hf_result['response'])
                result['generated_code'] = code_data
                result['explanation'] = "Código gerado via Hugging Face local."
                
                return result
        
        # Se tudo falhar, usa template básico
        logger.warning("Todos os modelos falharam, usando template básico")
        result['raw_response'] = "Fallback para template básico"
        result['generated_code'] = "# Código gerado por template básico\nprint('Implementação pendente')"
        result['explanation'] = "Código gerado usando template básico devido a falha nos modelos"
        result['used_model'] = 'template_fallback'
        result['confidence_score'] = 0.3
        
        return result

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
        if not HAS_BS4:
            return ValidationResult(status=ValidationStatus.SKIPPED, message="BeautifulSoup não disponível para validação de consistência.", validator_name="Consistency")
        
        soup = BeautifulSoup(html, 'html.parser')
        selectors = re.findall(r'page\.(?:locator|click|fill)\( مثلاً"|\["\\](.*?)\["\\]\)', code)
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

# --- Gerenciador de Templates de Código ---
class CodeTemplateManager:
    """Gerenciador de templates de código reutilizáveis."""
    
    def __init__(self):
        self.templates = {
            'rpa_click_template': '''
# Template para clique em elemento
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

try:
    element = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.{selector_type}, "{selector_value}"))
    )
    element.click()
    logger.info(f"Clicou no elemento: {selector_value}")
except Exception as e:
    logger.error(f"Erro ao clicar no elemento: {e}")
    raise
''',
            
            'rpa_type_template': '''
# Template para digitação em campo
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.{selector_type}, "{selector_value}"))
    )
    element.clear()
    element.send_keys("{text_to_type}")
    logger.info(f"Digitou no campo: {selector_value}")
except Exception as e:
    logger.error(f"Erro ao digitar no campo: {e}")
    raise
''',
            
            'error_handling_template': '''
# Template para tratamento de erro
import time
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def {function_name}():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            {main_code}
            return True
        except (TimeoutException, NoSuchElementException) as e:
            logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logger.error(f"Todas as tentativas falharam para: {function_name}")
                raise
        except Exception as e:
            logger.error(f"Erro inesperado em {function_name}: {e}")
            raise
    return False
''',
            
            'data_extraction_template': '''
# Template para extração de dados
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

try:
    elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.{selector_type}, "{selector_value}"))
    )
    
    extracted_data = []
    for element in elements:
        data = {{
            'text': element.text.strip(),
            'value': element.get_attribute('value') if element.get_attribute('value') else '',
            'href': element.get_attribute('href') if element.get_attribute('href') else ''
        }}
        extracted_data.append(data)
    
    logger.info(f"Extraiu {len(extracted_data)} elementos")
    return extracted_data
    
except Exception as e:
    logger.error(f"Erro na extração de dados: {e}")
    return []
'''
        }
    
    def get_template(self, template_type: str) -> Optional[str]:
        """Retorna um template de código específico."""
        return self.templates.get(template_type)
    
    def apply_template(self, template: str, data: Dict[str, Any]) -> str:
        """Aplica dados ao template substituindo placeholders."""
        try:
            return template.format(**data)
        except KeyError as e:
            logger.error(f"Placeholder não encontrado no template: {e}")
            return template
        except Exception as e:
            logger.error(f"Erro ao aplicar template: {e}")
            return template
    
    def list_templates(self) -> List[str]:
        """Lista todos os templates disponíveis."""
        return list(self.templates.keys())

# --- Framework Ético Simplificado (para integração) ---
@dataclass
class EthicalValidationResult:
    """Resultado da validação ética."""
    is_approved: bool
    reason: str
    risk_score: float
    requires_confirmation: bool = False

class SimpleEthicalFramework:
    """Framework ético simplificado para validação de ações."""
    
    def __init__(self):
        self.risk_keywords = [
            'delete', 'remove', 'drop', 'truncate', 'rm -rf',
            'format', 'destroy', 'wipe', 'system', 'admin',
            'password', 'private', 'secret', 'token'
        ]
        
        self.high_risk_actions = [
            'file_deletion', 'system_modification', 'data_destruction',
            'privilege_escalation', 'network_access'
        ]
    
    def validate_action(self, action_context: Dict[str, Any]) -> EthicalValidationResult:
        """Valida ação do ponto de vista ético."""
        risk_score = 1.0
        warnings = []
        
        # Verifica palavras-chave de risco
        description_lower = action_context.get('description', '').lower()
        for keyword in self.risk_keywords:
            if keyword in description_lower:
                risk_score *= 0.7
                warnings.append(f"Palavra-chave de risco detectada: {keyword}")
        
        # Verifica tipo de ação
        action_type = action_context.get('action_type', '')
        if action_type in self.high_risk_actions:
            risk_score *= 0.5
            warnings.append(f"Tipo de ação de alto risco: {action_type}")
        
        # Verifica parâmetros
        if 'generated_code' in action_context.get('parameters', {}):
            code = action_context['parameters']['generated_code']
            for keyword in self.risk_keywords:
                if keyword in code.lower():
                    risk_score *= 0.6
                    warnings.append(f"Código contém operação de risco: {keyword}")
        
        # Determina aprovação
        is_approved = risk_score > 0.5
        requires_confirmation = 0.3 < risk_score <= 0.7
        
        reason = "Ação aprovada" if is_approved else "Ação rejeitada por riscos éticos"
        if warnings:
            reason += f". Avisos: {'; '.join(warnings)}"
        
        return EthicalValidationResult(
            is_approved=is_approved,
            reason=reason,
            risk_score=risk_score,
            requires_confirmation=requires_confirmation
        )

# --- Sistema de Feedback Metacognitivo ---
class ACCMetacognitiveFeedback:
    """Sistema de feedback metacognitivo para o ACC."""
    
    def __init__(self):
        self.generation_history = []
        self.success_patterns = {}
        self.failure_patterns = {}
    
    def record_generation_attempt(self, context: Dict[str, Any], 
                                result: Dict[str, Any]) -> None:
        """Registra tentativa de geração para aprendizado."""
        record = {
            'timestamp': time.time(),
            'context_hash': hash(str(sorted(context.items()))),
            'success': result.get('success', False),
            'confidence': result.get('confidence', 0.0),
            'used_model': result.get('used_model', 'unknown'),
            'quality_score': result.get('quality_score', 0.0),
            'context_type': self._classify_context(context),
            'outcome_type': self._classify_outcome(context.get('desired_outcome', ''))
        }
        
        self.generation_history.append(record)
        self._update_patterns(record)
    
    def _classify_context(self, context: Dict[str, Any]) -> str:
        """Classifica tipo de contexto."""
        if 'error_message' in context:
            return 'error_correction'
        elif 'html_snapshot' in context:
            return 'web_automation'
        else:
            return 'general'
    
    def _classify_outcome(self, desired_outcome: str) -> str:
        """Classifica tipo de resultado desejado."""
        outcome_lower = desired_outcome.lower()
        if 'click' in outcome_lower:
            return 'click_action'
        elif 'type' in outcome_lower or 'digitar' in outcome_lower:
            return 'input_action'
        elif 'extract' in outcome_lower:
            return 'data_extraction'
        else:
            return 'other'
    
    def _update_patterns(self, record: Dict[str, Any]) -> None:
        """Atualiza padrões de sucesso/falha."""
        pattern_key = f"{record['context_type']}_{record['outcome_type']}"
        
        if record['success']:
            if pattern_key not in self.success_patterns:
                self.success_patterns[pattern_key] = []
            self.success_patterns[pattern_key].append(record)
        else:
            if pattern_key not in self.failure_patterns:
                self.failure_patterns[pattern_key] = []
            self.failure_patterns[pattern_key].append(record)
    
    def get_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fornece recomendações baseadas no histórico."""
        context_type = self._classify_context(context)
        outcome_type = self._classify_outcome(context.get('desired_outcome', ''))
        pattern_key = f"{context_type}_{outcome_type}"
        
        recommendations = {
            'suggested_model': 'auto',
            'expected_confidence': 0.5,
            'risk_factors': [],
            'success_probability': 0.5
        }
        
        # Análise de padrões de sucesso
        if pattern_key in self.success_patterns:
            successes = self.success_patterns[pattern_key]
            if successes:
                # Modelo mais bem-sucedido
                model_counts = {}
                for success in successes[-10:]:  # Últimos 10 sucessos
                    model = success['used_model']
                    model_counts[model] = model_counts.get(model, 0) + 1
                
                if model_counts:
                    recommendations['suggested_model'] = max(model_counts, key=model_counts.get)
                
                # Confiança média
                avg_confidence = sum(s['confidence'] for s in successes[-10:]) / len(successes[-10:])
                recommendations['expected_confidence'] = avg_confidence
        
        # Análise de padrões de falha
        if pattern_key in self.failure_patterns:
            failures = self.failure_patterns[pattern_key]
            total_attempts = len(self.success_patterns.get(pattern_key, [])) + len(failures)
            recommendations['success_probability'] = len(self.success_patterns.get(pattern_key, [])) / max(total_attempts, 1)
            
            # Identifica fatores de risco comuns
            for failure in failures[-5:]:  # Últimas 5 falhas
                if failure.get('ethical_issue'):
                    recommendations['risk_factors'].append('ethical_concerns')
                if failure.get('validation_failed'):
                    recommendations['risk_factors'].append('code_quality')
        
        return recommendations

# --- Sistema de Execução Simulada para Teste ---
class ExecutionSandbox:
    """Sandbox simulado para execução de código gerado."""
    
    def __init__(self):
        self.execution_history = []
        self.logger = logger
    
    async def simulate_execution(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simula execução do código gerado."""
        self.logger.info("=== SIMULAÇÃO DE EXECUÇÃO ===")
        self.logger.info(f"Código a ser executado:\n{code}")
        
        # Análise básica do código para simulação
        result = {
            'success': True,
            'execution_time': 2.5,  # Simulado
            'output': '',
            'errors': [],
            'side_effects': []
        }
        
        # Simula diferentes cenários baseados no código
        if 'click()' in code:
            result['output'] = 'Elemento clicado com sucesso'
            result['side_effects'].append('page_navigation')
        elif 'send_keys(' in code:
            result['output'] = 'Texto digitado no campo'
            result['side_effects'].append('form_input')
        elif 'find_element' in code and 'TimeoutException' in code:
            result['output'] = 'Elemento encontrado após aguardar'
        
        # Simula falhas ocasionais
        import random
        if random.random() < 0.1:  # 10% de chance de falha
            result['success'] = False
            result['errors'].append('Elemento não encontrado na simulação')
        
        # Registra histórico
        execution_record = {
            'timestamp': time.time(),
            'code_hash': hash(code),
            'context': context,
            'result': result
        }
        self.execution_history.append(execution_record)
        
        self.logger.info(f"Resultado da simulação: {result}")
        return result

# --- Gerador de Código Principal ---

class AdvancedCodeGenerator:
    """Orquestra o processo de geração e validação de código."""

    def __init__(self, llm_integration: Optional[LLMIntegration] = None, max_iterations: int = 2,
                 ethical_framework: Optional[SimpleEthicalFramework] = None,
                 metacognitive_feedback: Optional[ACCMetacognitiveFeedback] = None,
                 execution_sandbox: Optional[ExecutionSandbox] = None):
        self.llm = llm_integration or LLMIntegration()
        self.validator = CodeValidator()
        self.max_iterations = max_iterations
        self.ethical_framework = ethical_framework or SimpleEthicalFramework()
        self.metacognitive_feedback = metacognitive_feedback or ACCMetacognitiveFeedback()
        self.execution_sandbox = execution_sandbox or ExecutionSandbox()
        self.template_manager = CodeTemplateManager()

    def _build_optimized_prompt(self, context: ProblemContext, issues: List[str] = None) -> str:
        """Constrói um prompt otimizado para o LLM."""
        prompt = f"""
Você é um especialista em automação web Python com Playwright. Gere apenas o bloco de código Python solicitado, sem explicações adicionais.

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
            generated_code_response = await self.llm.generate_text_with_code(prompt)
            generated_code = generated_code_response.get('generated_code', '')

            if not generated_code:
                logger.warning("Falha na geração de código pelo LLM.")
                continue

            validation_results = self.validator.validate_all(generated_code, context)
            
            # Validação Ética
            ethical_context = {
                "action_type": "code_generation",
                "description": context.description,
                "parameters": {"generated_code": generated_code}
            }
            ethical_validation = self.ethical_framework.validate_action(ethical_context)
            
            if not ethical_validation.is_approved:
                logger.warning(f"Código reprovado pela ética: {ethical_validation.reason}")
                validation_results.append(ValidationResult(
                    status=ValidationStatus.FAILED, 
                    message=f"Reprovado pela ética: {ethical_validation.reason}", 
                    validator_name="Ethical"
                ))

            # Simulação de Execução
            simulation_result = await self.execution_sandbox.simulate_execution(generated_code, context)
            if not simulation_result['success']:
                logger.warning(f"Simulação de execução falhou: {simulation_result['errors']}")
                validation_results.append(ValidationResult(
                    status=ValidationStatus.FAILED, 
                    message=f"Simulação falhou: {simulation_result['errors']}", 
                    validator_name="Simulation"
                ))

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
        
        # Registrar feedback metacognitivo
        self.metacognitive_feedback.record_generation_attempt(context.__dict__, best_solution.__dict__)

        if not best_solution.code:
             return GeneratedSolution(code="# Falha ao gerar código após todas as iterações.", validation_results=[
                ValidationResult(status=ValidationStatus.FAILED, message="Nenhum código foi gerado.", validator_name="Generator")
             ])

        logger.info(f"Geração concluída. Score final da melhor solução: {best_solution.quality_score:.2f}")
        return best_solution

# --- Fachada para Integração Simplificada ---

class AutoCodeConstructorFacade:
    """Ponto de entrada simplificado para o sistema de geração de código."""

    def __init__(self, llm_integration: Optional[LLMIntegration] = None, max_iterations: int = 2,
                 ethical_framework: Optional[SimpleEthicalFramework] = None,
                 metacognitive_feedback: Optional[ACCMetacognitiveFeedback] = None,
                 execution_sandbox: Optional[ExecutionSandbox] = None):
        try:
            self.generator = AdvancedCodeGenerator(
                llm_integration=llm_integration or LLMIntegration(),
                max_iterations=max_iterations,
                ethical_framework=ethical_framework or SimpleEthicalFramework(),
                metacognitive_feedback=metacognitive_feedback or ACCMetacognitiveFeedback(),
                execution_sandbox=execution_sandbox or ExecutionSandbox()
            )
            self.is_ready = True
            logger.info("Fachada do Construtor de Código inicializada com sucesso.")
        except Exception as e:
            self.generator = None
            self.is_ready = False
            logger.critical(f"Falha ao inicializar a fachada do construtor de código: {e}")

    async def generate_automation_code(self, error_context: str, html_content: str, target_action: str, **kwargs) -> Dict[str, Any]:
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
            target_url=kwargs.get('target_url', "URL não especificada no contexto do erro"),
            existing_code=kwargs.get('existing_code')
        )
        
        try:
            solution = await self.generator.generate_solution(context)
            
            if solution.is_valid:
                explanation = "O script gerado tenta corrigir o problema com base no contexto fornecido. Ele usa esperas explícitas e seletores robustos para maior confiabilidade."
                return {
                    "success": True,
                    "code": solution.code,
                    "explanation": explanation,
                    "quality_score": solution.quality_score,
                    "validation_results": [asdict(r) for r in solution.validation_results]
                }
            else:
                failed_checks = [f"{r.validator_name}: {r.message}" for r in solution.validation_results if r.status == ValidationStatus.FAILED]
                return {
                    "success": False,
                    "error": "O código gerado não passou nas validações críticas.",
                    "details": {
                        "failed_checks": failed_checks,
                        "generated_code_attempt": solution.code,
                        "validation_results": [asdict(r) for r in solution.validation_results]
                    }
                }
        except Exception as e:
            logger.error(f"Erro crítico no processo de geração de código: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Uma exceção interna ocorreu durante a geração do código: {e}"
            }

# --- Bloco de Demonstração (para testes) ---
async def main_demo():
    """Função de demonstração para testar o módulo de forma independente."""
    logger.info("--- INICIANDO DEMONSTRAÇÃO DO AUTO CONSTRUTOR DE CÓDIGO UNIFICADO ---")
    
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
        print("\n--- RESULTADOS DA VALIDAÇÃO ---")
        for res in resultado.get("validation_results", []):
            print(f"  - {res['validator_name']}: {res['status'].value} - {res['message']}")

    else:
        print("❌ Falha na geração de código.")
        print(f"Erro: {resultado.get('error')}")
        if "details" in resultado:
            print("Detalhes:", json.dumps(resultado["details"], indent=2))
            
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main_demo())