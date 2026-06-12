# atena_motor_voz.py
# Versão 2.0 - Motor de Voz Refatorado para Integração
import asyncio
import logging
import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import numpy as np

# --- Dependências de IA ---
try:
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch/Transformers não disponível. A transcrição de voz (Whisper) está desabilitada.")

# --- Dependências de Análise de Áudio ---
try:
    import librosa
    import difflib
    HAS_SPEECH_ANALYSIS_LIBS = True
except ImportError:
    HAS_SPEECH_ANALYSIS_LIBS = False
    logging.warning("Librosa/difflib não encontradas. Análise fonética limitada.")

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s')
logger = logging.getLogger("AtenaVoiceMotor")


# --- Estruturas de Dados e Classes de Perfil ---

class TipoDisturbio:
    """Enumeração simplificada dos tipos de distúrbios da fala."""
    DISTONIA_OROMANDIBULAR = "Distonia Oromandibular"
    NENHUM = "Nenhum"

@dataclass
class PerfilDeFala:
    """Representa o perfil clínico do usuário para guiar a correção."""
    disturbio_primario: str = TipoDisturbio.NENHUM
    intensidade: float = 0.5

@dataclass
class EntradaLexico:
    """Estrutura para cada mapeamento no léxico fonético."""
    transcricao_crua: str
    texto_confirmado: str
    contagem_confirmacao: int = 1
    ultimo_uso: str = field(default_factory=lambda: datetime.now().isoformat())

    def atualizar_uso(self):
        self.ultimo_uso = datetime.now().isoformat()

# --- Componentes Principais ---

class AnalisadorFonetico:
    """Módulo que calcula a similaridade fonética entre textos."""
    def calcular_distancia_fonetica(self, s1: str, s2: str) -> float:
        if not HAS_SPEECH_ANALYSIS_LIBS:
            return 0.0
        return difflib.SequenceMatcher(None, s1, s2).ratio()

class LexicoFoneticoManager:
    """Gerencia o léxico de correções fonéticas personalizadas."""
    def __init__(self, arquivo_lexico: Path, perfil_fala: PerfilDeFala):
        self.arquivo_lexico = arquivo_lexico
        self.perfil_fala = perfil_fala
        self.lexico: Dict[str, EntradaLexico] = {}
        self.analisador_fonetico = AnalisadorFonetico()
        self._carregar_lexico()

    def _carregar_lexico(self):
        self.arquivo_lexico.parent.mkdir(parents=True, exist_ok=True)
        if self.arquivo_lexico.exists():
            try:
                with open(self.arquivo_lexico, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                for chave, entrada_dict in dados.items():
                    self.lexico[chave] = EntradaLexico(**entrada_dict)
                logger.info(f"Léxico Fonético carregado com {len(self.lexico)} entradas de '{self.arquivo_lexico}'.")
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Erro ao carregar léxico de '{self.arquivo_lexico}': {e}. Iniciando com léxico vazio.")
                self.lexico = {}

    def _salvar_lexico(self):
        try:
            with open(self.arquivo_lexico, 'w', encoding='utf-8') as f:
                # Usando asdict para converter dataclasses em dicionários para serialização JSON
                json.dump({k: asdict(v) for k, v in self.lexico.items()}, f, ensure_ascii=False, indent=4)
            logger.info(f"Léxico Fonético salvo com {len(self.lexico)} entradas em '{self.arquivo_lexico}'.")
        except Exception as e:
            logger.error(f"Falha ao salvar o léxico em '{self.arquivo_lexico}': {e}", exc_info=True)

    def aprender_mapeamento(self, transcricao_crua: str, texto_confirmado: str):
        """Aprende ou reforça um mapeamento de uma transcrição errada para uma correta."""
        chave = self._normalizar_texto(transcricao_crua)
        texto_confirmado_norm = self._normalizar_texto(texto_confirmado)

        if not chave or not texto_confirmado_norm:
            logger.warning("Tentativa de aprender mapeamento com texto vazio. Ignorando.")
            return

        if chave in self.lexico:
            entrada = self.lexico[chave]
            entrada.texto_confirmado = texto_confirmado_norm # Sempre atualiza para o mais recente
            entrada.contagem_confirmacao += 1
            entrada.atualizar_uso()
            logger.info(f"Reforço de mapeamento: '{chave}' -> '{texto_confirmado_norm}' (x{entrada.contagem_confirmacao})")
        else:
            logger.info(f"Novo mapeamento aprendido: '{chave}' -> '{texto_confirmado_norm}'")
            nova_entrada = EntradaLexico(
                transcricao_crua=chave,
                texto_confirmado=texto_confirmado_norm,
            )
            self.lexico[chave] = nova_entrada

        self._salvar_lexico()

    def corrigir_transcricao(self, texto_transcrito: str) -> str:
        """Aplica a correção do léxico a uma nova transcrição."""
        texto_normalizado = self._normalizar_texto(texto_transcrito)

        # 1. Busca Direta
        if texto_normalizado in self.lexico:
            entrada = self.lexico[texto_normalizado]
            entrada.atualizar_uso()
            logger.info(f"Correção por Léxico (Busca Direta): '{texto_normalizado}' -> '{entrada.texto_confirmado}'")
            return entrada.texto_confirmado

        # 2. Busca por Similaridade
        melhor_match, similaridade = self._buscar_por_similaridade(texto_normalizado)
        if melhor_match and similaridade > 0.85: # Limiar de confiança alto
            melhor_match.atualizar_uso()
            logger.info(f"Correção por Léxico (Similaridade > 85%): '{texto_normalizado}' -> '{melhor_match.texto_confirmado}'")
            return melhor_match.texto_confirmado

        logger.info(f"Nenhuma correção forte encontrada no léxico para: '{texto_transcrito}'")
        return texto_transcrito

    def _normalizar_texto(self, texto: str) -> str:
        return re.sub(r'\s+', ' ', texto.lower().strip())

    def _buscar_por_similaridade(self, texto: str) -> (Optional[EntradaLexico], float):
        if not self.lexico:
            return None, 0.0
        
        melhor_entrada = None
        maior_similaridade = 0.0

        for entrada in self.lexico.values():
            similaridade = self.analisador_fonetico.calcular_distancia_fonetica(texto, entrada.transcricao_crua)
            if similaridade > maior_similaridade:
                maior_similaridade = similaridade
                melhor_entrada = entrada

        return melhor_entrada, maior_similaridade

class AdaptiveSpeechProcessor:
    """Encapsula o modelo Whisper para transcrição de áudio."""
    def __init__(self, sample_rate: int = 16000):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch/Transformers não estão instalados. O processador de fala não pode funcionar.")
        
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Usando dispositivo '{self.device}' para o modelo Whisper.")

        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(self.device)
            logger.info("Modelo Whisper (openai/whisper-large-v3) carregado com sucesso.")
        except Exception as e:
            logger.critical(f"Falha ao carregar o modelo Whisper: {e}", exc_info=True)
            raise

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcreve bytes de áudio para texto usando o modelo Whisper."""
        if not HAS_SPEECH_ANALYSIS_LIBS:
            return "Erro: bibliotecas de análise de áudio não instaladas."
        
        try:
            # Converte bytes para um array numpy float32
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Reamostra se a taxa de amostragem for diferente da esperada pelo Whisper (16kHz)
            # Assumindo que o áudio de entrada é 44.1kHz, uma taxa comum. Ajustar se necessário.
            # NOTA: A taxa de amostragem original precisa ser conhecida. Usando 44100 como um padrão comum.
            current_sample_rate = 44100
            if current_sample_rate != self.sample_rate:
                audio_np = librosa.resample(audio_np, orig_sr=current_sample_rate, target_sr=self.sample_rate)

            # Processa o áudio e envia para o modelo
            input_features = self.processor(audio_np, sampling_rate=self.sample_rate, return_tensors="pt").input_features
            predicted_ids = await asyncio.to_thread(self.model.generate, input_features.to(self.device))
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            logger.info(f"Transcrição do Whisper: '{transcription}'")
            return transcription.strip()

        except Exception as e:
            logger.error(f"Erro durante a transcrição com Whisper: {e}", exc_info=True)
            return f"Erro na transcrição: {e}"


# --- CLASSE PRINCIPAL DO MOTOR DE VOZ ---

class AtenaVoiceMotor:
    """
    Motor de Voz Unificado da Atena (v2.0).
    Orquestra a transcrição de áudio e a correção fonética.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        
        # Configuração do perfil de fala e léxico
        perfil_path = Path(config.get("lexico_path", "./memoria_do_usuario/lexico_fonetico.json"))
        perfil_fala = PerfilDeFala(disturbio_primario=TipoDisturbio.DISTONIA_OROMANDIBULAR)
        
        self.lexico_manager = LexicoFoneticoManager(arquivo_lexico=perfil_path, perfil_fala=perfil_fala)
        
        # Inicializa o processador de fala (Whisper)
        try:
            self.speech_processor = AdaptiveSpeechProcessor()
        except RuntimeError as e:
            logger.critical(f"Não foi possível inicializar o AtenaVoiceMotor: {e}")
            self.speech_processor = None
        
        logger.info("🎤 Atena Voice Motor v2.0 (Refatorado) inicializado.")

    async def transcribe_audio(self, audio_bytes: bytes) -> Dict[str, str]:
        """
        Processo completo: transcreve o áudio e depois aplica a correção do léxico.
        Retorna tanto a transcrição crua quanto a corrigida.
        """
        if not self.speech_processor:
            return {"raw_text": "", "corrected_text": "Erro: Processador de fala não inicializado."}

        # 1. Transcrição com Whisper
        raw_transcription = await self.speech_processor.transcribe(audio_bytes)
        
        # 2. Correção com o Léxico Fonético
        corrected_transcription = self.lexico_manager.corrigir_transcricao(raw_transcription)
        
        return {
            "raw_text": raw_transcription,
            "corrected_text": corrected_transcription
        }

    def learn_correction(self, raw_text: str, corrected_text: str):
        """
        Expõe o método de aprendizado do léxico para a API.
        """
        self.lexico_manager.aprender_mapeamento(raw_text, corrected_text)
        return {"status": "success", "message": f"Mapeamento de '{raw_text}' para '{corrected_text}' aprendido."}

    def shutdown(self):
        """Finaliza componentes se necessário."""
        self.lexico_manager._salvar_lexico()
        logger.info("🎤 Atena Voice Motor finalizado.")