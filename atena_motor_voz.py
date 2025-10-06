# atena_motor_voz.py
# Versão 1.0 - Módulo Unificado de Fala e Voz

import asyncio
import hashlib
import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from collections import defaultdict, OrderedDict
import json
import re
from enum import Enum

# Dependências principais
import pygame
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

# Dependências de NLP e análise
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK não disponível. Funcionalidades de NLP limitadas.")

# Dependências de IA avançada (para NLPProcessor)
try:
    import torch
    from transformers import pipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch/Transformers não disponível. IA avançada desabilitada.")

# Dependências do Edge-TTS
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logging.error("Edge-TTS não disponível. Instale com: pip install edge-tts")

# Dependências para fala adaptativa e léxico fonético
try:
    import librosa
    import scipy.signal as signal
    from scipy.interpolate import interp1d
    HAS_SPEECH_ANALYSIS_LIBS = True
except ImportError:
    HAS_SPEECH_ANALYSIS_LIBS = False
    logging.warning("Bibliotecas de análise de fala (librosa, scipy) não encontradas. Funcionalidades adaptativas limitadas.")

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AtenaVoiceEngine")

# --- Estruturas de Dados Comuns ---

@dataclass
class AudioMetrics:
    """Métricas de qualidade e performance do áudio gerado."""
    duration: float = 0.0
    sample_rate: int = 22050
    bit_depth: int = 16
    channels: int = 1
    snr_db: float = 0.0  # Signal-to-Noise Ratio
    generation_time: float = 0.0
    cache_hit: bool = False
    quality_score: float = 0.0  # 0-1, calculado por IA
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte métricas para dicionário."""
        return {
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'bit_depth': self.bit_depth,
            'channels': self.channels,
            'snr_db': self.snr_db,
            'generation_time': self.generation_time,
            'cache_hit': self.cache_hit,
            'quality_score': self.quality_score
        }

@dataclass
class CacheEntry:
    """Entrada do cache com metadados avançados."""
    audio_data: bytes
    metadata: AudioMetrics
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    text_hash: str = ""
    voice_model_hash: str = ""
    priority_score: float = 0.0  # Para algoritmo de eviction
    
    def update_access(self):
        """Atualiza estatísticas de acesso."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        # Atualiza score de prioridade (mais acessos recentes = maior prioridade)
        time_factor = 1.0 - (datetime.now() - self.last_accessed).days / 30.0
        self.priority_score = self.access_count * max(0.1, time_factor)

# --- Gerenciador de Cache Inteligente (do atena_fala.py) ---
class SmartCacheManager:
    """
    Gerenciador de cache inteligente com 4 camadas:
    1. Memória RAM (ultra-rápido)
    2. Cache de disco SSD (rápido)
    3. Cache comprimido (economia de espaço)
    4. Cache distribuído (futuro: Redis/Memcached)
    """
    
    def __init__(self, base_dir: Path, max_memory_mb: int = 100, 
                 max_disk_mb: int = 1000):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_size = max_memory_mb * 1024 * 1024  # bytes
        self.max_disk_size = max_disk_mb * 1024 * 1024      # bytes
        
        self.memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory_size = 0
        
        self.disk_cache_dir = self.base_dir / "disk_cache"
        self.disk_cache_dir.mkdir(exist_ok=True)
        
        self.cache_metadata_file = self.base_dir / "cache_metadata.json"
        self.disk_metadata: Dict[str, Dict] = {}
        
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        self.lock = threading.RLock()
        
        self._load_disk_metadata()
        self._start_cleanup_thread()
    
    def _generate_cache_key(self, text: str, voice_model: str = "default") -> str:
        combined = f"{text}:{voice_model}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _load_disk_metadata(self):
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r', encoding='utf-8') as f:
                    self.disk_metadata = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Erro ao carregar metadados do cache: {e}. Criando um novo.")
                self.disk_metadata = {}
    
    def _save_disk_metadata(self):
        try:
            with open(self.cache_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.disk_metadata, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Erro ao salvar metadados do cache: {e}")
    
    def _start_cleanup_thread(self):
        def cleanup_worker():
            while True:
                time.sleep(300)  # Executa a cada 5 minutos
                try:
                    self._cleanup_expired_entries()
                except Exception as e:
                    logger.error(f"Erro na limpeza automática do cache: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_entries(self):
        with self.lock:
            now = datetime.now()
            expired_keys = []
            
            for key, entry in self.memory_cache.items():
                if (now - entry.last_accessed).days > 7:  # 7 dias sem acesso
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            for key, metadata in list(self.disk_metadata.items()):
                last_accessed = datetime.fromisoformat(metadata.get('last_accessed', '2000-01-01'))
                if (now - last_accessed).days > 30:  # 30 dias sem acesso
                    cache_file = self.disk_cache_dir / f"{key}.pkl"
                    if cache_file.exists():
                        cache_file.unlink()
                    del self.disk_metadata[key]
            
            self._save_disk_metadata()
    
    def get(self, text: str, voice_model: str = "default") -> Optional[CacheEntry]:
        with self.lock:
            self.stats['total_requests'] += 1
            cache_key = self._generate_cache_key(text, voice_model)
            
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                entry.update_access()
                self.memory_cache.move_to_end(cache_key)
                self.stats['memory_hits'] += 1
                return entry
            
            if cache_key in self.disk_metadata:
                cache_file = self.disk_cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            entry = pickle.load(f)
                        entry.update_access()
                        
                        self._add_to_memory_cache(cache_key, entry)
                        
                        self.stats['disk_hits'] += 1
                        return entry
                    except Exception as e:
                        logger.error(f"Erro ao carregar cache do disco: {e}")
                        cache_file.unlink()
                        del self.disk_metadata[cache_key]
            
            self.stats['misses'] += 1
            return None
    
    def put(self, text: str, audio_data: bytes, metadata: AudioMetrics, 
            voice_model: str = "default"):
        with self.lock:
            cache_key = self._generate_cache_key(text, voice_model)
            
            entry = CacheEntry(
                audio_data=audio_data,
                metadata=metadata,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                text_hash=cache_key,
                voice_model_hash=hashlib.sha256(voice_model.encode()).hexdigest()
            )
            
            self._add_to_memory_cache(cache_key, entry)
            self._save_to_disk_cache(cache_key, entry)
    
    def _add_to_memory_cache(self, key: str, entry: CacheEntry):
        entry_size = len(entry.audio_data)
        
        while (self.current_memory_size + entry_size > self.max_memory_size 
               and len(self.memory_cache) > 0):
            self._evict_from_memory()
        
        self.memory_cache[key] = entry
        self.current_memory_size += entry_size
        
        self.memory_cache.move_to_end(key)
    
    def _evict_from_memory(self):
        if not self.memory_cache:
            return
        
        min_score = float('inf')
        key_to_evict = None
        
        for key, entry in self.memory_cache.items():
            if entry.priority_score < min_score:
                min_score = entry.priority_score
                key_to_evict = key
        
        if key_to_evict:
            entry = self.memory_cache.pop(key_to_evict)
            self.current_memory_size -= len(entry.audio_data)
            self.stats['evictions'] += 1
    
    def _save_to_disk_cache(self, key: str, entry: CacheEntry):
        try:
            cache_file = self.disk_cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
            
            self.disk_metadata[key] = {
                'created_at': entry.created_at.isoformat(),
                'last_accessed': entry.last_accessed.isoformat(),
                'access_count': entry.access_count,
                'size': len(entry.audio_data),
                'text_hash': entry.text_hash,
                'voice_model_hash': entry.voice_model_hash
            }
            
            if len(self.disk_metadata) % 10 == 0:
                self._save_disk_metadata()
                
        except Exception as e:
            logger.error(f"Erro ao salvar cache no disco: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            hit_rate = 0.0
            if self.stats['total_requests'] > 0:
                total_hits = self.stats['memory_hits'] + self.stats['disk_hits']
                hit_rate = total_hits / self.stats['total_requests']
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'memory_entries': len(self.memory_cache),
                'disk_entries': len(self.disk_metadata),
                'memory_size_mb': self.current_memory_size / (1024 * 1024),
                'cache_efficiency': self._calculate_cache_efficiency()
            }
    
    def _calculate_cache_efficiency(self) -> float:
        if not self.memory_cache:
            return 0.0
        
        total_score = sum(entry.priority_score for entry in self.memory_cache.values())
        avg_score = total_score / len(self.memory_cache)
        
        return min(1.0, avg_score / 10.0)

# --- Processador de Áudio (do atena_voice.py) ---
class AudioProcessor:
    """Processador de áudio para manipulação e concatenação."""
    def __init__(self):
        self.default_word_pause_ms = 120
        logger.info("AudioProcessor inicializado.")

    def concatenate_words(self, word_audios: List[AudioSegment]) -> Optional[AudioSegment]:
        if not PYDUB_AVAILABLE or not word_audios:
            return AudioSegment.empty() if PYDUB_AVAILABLE else None
        
        pause = AudioSegment.silent(duration=self.default_word_pause_ms)
        full_audio = AudioSegment.empty()
        for i, word_audio in enumerate(word_audios):
            full_audio += word_audio
            if i < len(word_audios) - 1:
                full_audio += pause
        return full_audio
        
    def play_audio_async(self, audio: AudioSegment):
        """Reproduz áudio em uma thread separada para não bloquear."""
        if PYDUB_AVAILABLE and audio:
            try:
                playback_thread = threading.Thread(target=play, args=(audio,))
                playback_thread.daemon = True
                playback_thread.start()
                logger.debug("Reprodução de áudio iniciada em thread separada.")
            except Exception as e:
                logger.error(f"Erro ao iniciar thread de reprodução: {e}")
        else:
            logger.warning("pydub não está disponível, reprodução de áudio pulada.")

# --- Processador de Linguagem Natural (do atena_fala.py) ---
class NLPProcessor:
    """Processador de linguagem natural para otimização de TTS."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if NLTK_AVAILABLE:
            self._initialize_nltk()
        
        if TORCH_AVAILABLE:
            self._initialize_transformers()
        
        self.normalization_patterns = {
            r'\b\d+\b': self._spell_number,
            r'https?://[^\s]+': 'link',
            r'[^\s]+@[^\s]+\.[^\s]+': 'email',
            r'\bDr\.': 'Doutor',
            r'\bSr\.': 'Senhor',
            r'\bSra\.': 'Senhora',
            r'&': 'e',
            r'@': 'arroba',
            r'#': 'hashtag',
        }
    
    def _initialize_nltk(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            self.logger.warning(f"Erro ao inicializar NLTK: {e}")
    
    def _initialize_transformers(self):
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="neuralmind/bert-base-portuguese-cased",
                device=-1  # CPU only
            )
            
            self.ner_pipeline = pipeline(
                "ner",
                model="neuralmind/bert-base-portuguese-cased",
                device=-1
            )
            
            self.logger.info("Modelos de transformers carregados com sucesso")
        except Exception as e:
            self.logger.warning(f"Erro ao carregar transformers: {e}")
            self.sentiment_analyzer = None
            self.ner_pipeline = None
    
    def _spell_number(self, match):
        num = int(match.group())
        numbers = {
            0: 'zero', 1: 'um', 2: 'dois', 3: 'três', 4: 'quatro',
            5: 'cinco', 6: 'seis', 7: 'sete', 8: 'oito', 9: 'nove',
            10: 'dez', 11: 'onze', 12: 'doze', 13: 'treze', 14: 'catorze',
            15: 'quinze', 16: 'dezesseis', 17: 'dezessete', 18: 'dezoito',
            19: 'dezenove', 20: 'vinte'
        }
        return numbers.get(num, str(num))
    
    def normalize_text(self, text: str) -> str:
        normalized = text.strip()
        
        for pattern, replacement in self.normalization_patterns.items():
            if callable(replacement):
                normalized = re.sub(pattern, replacement, normalized)
            else:
                normalized = re.sub(pattern, replacement, normalized)
        
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        if not self.sentiment_analyzer:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                'label': result['label'],
                'score': result['score'],
                'prosody_adjustment': self._get_prosody_adjustment(result)
            }
        except Exception as e:
            self.logger.error(f"Erro na análise de sentimento: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def _get_prosody_adjustment(self, sentiment: Dict) -> Dict[str, float]:
        adjustments = {
            'speed': 1.0,
            'pitch': 1.0,
            'volume': 1.0
        }
        
        if sentiment['label'] == 'POSITIVE':
            adjustments['speed'] = 1.1
            adjustments['pitch'] = 1.05
            adjustments['volume'] = 1.0
        elif sentiment['label'] == 'NEGATIVE':
            adjustments['speed'] = 0.9
            adjustments['pitch'] = 0.95
            adjustments['volume'] = 0.95
        
        return adjustments
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if not self.ner_pipeline:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            return [
                {
                    'word': entity['word'],
                    'label': entity['entity'],
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                }
                for entity in entities
            ]
        except Exception as e:
            self.logger.error(f"Erro na extração de entidades: {e}")
            return []
    
    def smart_sentence_split(self, text: str) -> List[str]:
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text, language='portuguese')
                return [s.strip() for s in sentences if s.strip()]
            except Exception as e:
                self.logger.warning(f"Erro no NLTK sentence split: {e}")
        
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

# --- Motor de Síntese de Fala Unificado ---
class UnifiedTTSEngine:
    """
    Sistema unificado de síntese de fala, utilizando Edge-TTS como backend principal.
    Integra cache inteligente e processamento NLP.
    """
    
    SUPPORTED_FORMATS = ['mp3', 'wav', 'ogg']
    DEFAULT_SAMPLE_RATE = 24000  # Edge-TTS usa 24kHz por padrão
    DEFAULT_BITRATE = 128000
    MAX_TEXT_LENGTH = 5000
    
    def __init__(self, 
                 voice: str = "pt-BR-FranciscaNeural",
                 cache_dir: str = "./atena_tts_cache",
                 max_cache_size_mb: int = 500,
                 audio_format: str = 'mp3'):
        
        self.voice = voice
        self.audio_format = audio_format.lower()
        self.cache_dir = Path(cache_dir)
        
        if self.audio_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Formato não suportado: {audio_format}")
        
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("Edge-TTS não está disponível. Instale com: pip install edge-tts")
        
        self.cache_manager = SmartCacheManager(
            self.cache_dir, 
            max_memory_mb=max_cache_size_mb // 5, 
            max_disk_mb=max_cache_size_mb
        )
        
        self.nlp_processor = NLPProcessor()
        self.audio_processor = AudioProcessor() # Do atena_voice.py
        
        self.model_hash = hashlib.sha256(self.voice.encode()).hexdigest()[:16]
        
        self._initialize_audio_system()
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.stats = {
            'total_generations': 0,
            'cache_hits': 0,
            'total_duration': 0.0,
            'avg_generation_time': 0.0,
            'quality_scores': []
        }
        
        logger.info(f"Sistema de síntese de fala unificado inicializado com sucesso usando a voz: {self.voice}")
    
    def _initialize_audio_system(self):
        try:
            pygame.mixer.pre_init(
                frequency=self.DEFAULT_SAMPLE_RATE,
                size=-16,
                channels=1)
            pygame.mixer.init()
            logger.info("Sistema de áudio inicializado")
        except Exception as e:
            logger.error(f"Erro ao inicializar áudio: {e}")
            raise
    
    def _validate_input(self, text: str) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Texto deve ser uma string não vazia")
        
        text = text.strip()
        
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(f"Texto truncado de {len(text)} para {self.MAX_TEXT_LENGTH} caracteres")
            text = text[:self.MAX_TEXT_LENGTH]
        
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text
    
    async def _generate_audio_engine(self, text: str) -> Tuple[bytes, AudioMetrics]:
        start_time = time.time()
        
        try:
            audio_buffer = BytesIO()
            async def generate():
                communicate = edge_tts.Communicate(text, self.voice)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_buffer.write(chunk["data"])

            await asyncio.run(generate())

            audio_data = audio_buffer.getvalue()
            generation_time = time.time() - start_time
            
            audio_segment = AudioSegment.from_file(BytesIO(audio_data), format="mp3")
            duration_seconds = len(audio_segment) / 1000.0

            metrics = AudioMetrics(
                duration=duration_seconds,
                sample_rate=audio_segment.frame_rate,
                bit_depth=audio_segment.sample_width * 8,
                channels=audio_segment.channels,
                generation_time=generation_time,
                quality_score=self._calculate_audio_quality(audio_segment.raw_data)
            )
            
            return audio_data, metrics
            
        except Exception as e:
            logger.error(f"Erro na síntese de áudio com Edge-TTS: {e}")
            raise

    
    def _calculate_audio_quality(self, audio_data: bytes) -> float:
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0.0
            
            signal_variation = np.std(audio_array) / (np.max(np.abs(audio_array)) + 1e-6)
            
            fft = np.fft.fft(audio_array)
            spectral_flatness = np.exp(np.mean(np.log(np.abs(fft) + 1e-6))) / (np.mean(np.abs(fft)) + 1e-6)
            
            duration_score = min(1.0, len(audio_array) / self.DEFAULT_SAMPLE_RATE / 10.0)
            
            quality_score = (signal_variation * 0.4 + spectral_flatness * 0.4 + duration_score * 0.2)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de qualidade: {e}")
            return 0.5
    
    def _apply_prosody_adjustments(self, audio_data: bytes, adjustments: Dict[str, float]) -> bytes:
        try:
            audio_segment = AudioSegment(
                audio_data,
                frame_rate=self.DEFAULT_SAMPLE_RATE,
                sample_width=2,  # 16-bit
                channels=1
            )
            
            if adjustments.get('speed', 1.0) != 1.0:
                speed_factor = adjustments['speed']
                audio_segment = audio_segment._spawn(
                    audio_segment.raw_data,
                    overrides={'frame_rate': int(audio_segment.frame_rate * speed_factor)}
                ).set_frame_rate(audio_segment.frame_rate)
            
            if adjustments.get('volume', 1.0) != 1.0:
                volume_change = 20 * np.log10(adjustments['volume'])  # dB
                audio_segment = audio_segment + volume_change
            
            return audio_segment.raw_data
            
        except Exception as e:
            logger.warning(f"Erro nos ajustes de prosódia: {e}")
            return audio_data
    
    def _convert_audio_format(self, audio_data: bytes, target_format: str) -> bytes:
        try:
            audio_segment = AudioSegment(
                audio_data,
                frame_rate=self.DEFAULT_SAMPLE_RATE,
                sample_width=2,  # 16-bit
                channels=1
            )
            
            output_buffer = BytesIO()
            
            if target_format == 'mp3':
                audio_segment.export(output_buffer, format='mp3', bitrate=f"{self.DEFAULT_BITRATE}")
            elif target_format == 'wav':
                audio_segment.export(output_buffer, format='wav')
            elif target_format == 'ogg':
                audio_segment.export(output_buffer, format='ogg')
            else:
                raise ValueError(f"Formato não suportado: {target_format}")
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Erro na conversão de formato: {e}")
            raise
    
    async def synthesize_async(self, text: str, apply_nlp: bool = True) -> Tuple[bytes, AudioMetrics]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.synthesize, 
            text, 
            apply_nlp
        )
    
    def synthesize(self, text: str, apply_nlp: bool = True) -> Tuple[bytes, AudioMetrics]:
        start_time = time.time()
        
        try:
            text = self._validate_input(text)
            
            processed_text = text
            prosody_adjustments = {'speed': 1.0, 'pitch': 1.0, 'volume': 1.0}
            
            if apply_nlp:
                processed_text = self.nlp_processor.normalize_text(text)
                sentiment = self.nlp_processor.analyze_sentiment(processed_text)
                prosody_adjustments = sentiment.get('prosody_adjustment', prosody_adjustments)
            
            cache_entry = self.cache_manager.get(processed_text, self.model_hash)
            if cache_entry:
                self.stats['cache_hits'] += 1
                cache_entry.metadata.cache_hit = True
                logger.info(f"Cache hit para texto: {text[:50]}...")
                return cache_entry.audio_data, cache_entry.metadata
            
            raw_audio_data, metrics = self._generate_audio_engine(processed_text)
            
            if any(v != 1.0 for v in prosody_adjustments.values()):
                raw_audio_data = self._apply_prosody_adjustments(raw_audio_data, prosody_adjustments)
            
            final_audio_data = self._convert_audio_format(raw_audio_data, self.audio_format)
            
            total_time = time.time() - start_time
            metrics.generation_time = total_time
            
            self.cache_manager.put(processed_text, final_audio_data, metrics, self.model_hash)
            
            self._update_stats(metrics)
            
            logger.info(f"Síntese concluída em {total_time:.2f}s - Qualidade: {metrics.quality_score:.2f}")
            
            return final_audio_data, metrics
            
        except Exception as e:
            logger.error(f"Erro na síntese: {e}")
            raise
    
    def _update_stats(self, metrics: AudioMetrics):
        self.stats['total_generations'] += 1
        self.stats['total_duration'] += metrics.duration
        self.stats['quality_scores'].append(metrics.quality_score)
        
        if self.stats['avg_generation_time'] == 0:
            self.stats['avg_generation_time'] = metrics.generation_time
        else:
            alpha = 0.1
            self.stats['avg_generation_time'] = (
                alpha * metrics.generation_time + 
                (1 - alpha) * self.stats['avg_generation_time']
            )
    
    def play_audio(self, audio_data: bytes):
        if PYDUB_AVAILABLE:
            try:
                temp_file = self.cache_dir / f"temp_audio_{int(time.time())}.{self.audio_format}"
                
                with open(temp_file, 'wb') as f:
                    f.write(audio_data)
                
                if self.audio_format == 'mp3':
                    pygame.mixer.music.load(str(temp_file))
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                else:
                    audio_segment = AudioSegment.from_file(str(temp_file))
                    play(audio_segment)
                
                temp_file.unlink()
                
            except Exception as e:
                logger.error(f"Erro na reprodução: {e}")
                raise
        else:
            logger.warning("pydub não está disponível, reprodução de áudio pulada.")
    
    def save_audio(self, audio_data: bytes, filepath: str):
        try:
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            logger.info(f"Áudio salvo em: {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar áudio: {e}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        cache_stats = self.cache_manager.get_stats()
        
        quality_avg = 0.0
        if self.stats['quality_scores']:
            quality_avg = sum(self.stats['quality_scores']) / len(self.stats['quality_scores'])
        
        return {
            'synthesis': {
                **self.stats,
                'avg_quality_score': quality_avg
            },
            'cache': cache_stats,
            'system': {
                'audio_format': self.audio_format,
                'model_hash': self.model_hash,
                'nlp_available': NLTK_AVAILABLE,
                'ai_available': TORCH_AVAILABLE,
                'uptime_hours': (time.time() - time.time()) / 3600 # Placeholder
            }
        }
    
    def batch_synthesize(self, texts: List[str], apply_nlp: bool = True) -> List[Tuple[bytes, AudioMetrics]]:
        results = []
        
        for text in texts:
            try:
                audio_data, metrics = self.synthesize(text, apply_nlp)
                results.append((audio_data, metrics))
            except Exception as e:
                self.logger.error(f"Erro na síntese em lote para '{text[:50]}...': {e}")
                results.append((b'', AudioMetrics()))
        
        return results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def shutdown(self):
        try:
            self.executor.shutdown(wait=True)
            self.cache_manager._save_disk_metadata()
            pygame.mixer.quit()
            logger.info("Sistema de síntese de fala finalizado com sucesso")
        except Exception as e:
            logger.error(f"Erro na finalização do sistema de fala: {e}")

# --- Analisador Clínico Fonético (do atena_lexico_fonetico.py) ---
class TipoDisturbio(Enum):
    """Enumeração dos tipos de distúrbios da fala modelados."""
    DISTONIA_LARINGEA = "Distonia Laríngea (Disfonia Espasmódica)"
    DISTONIA_CERVICAL = "Distonia Cervical"
    DISTONIA_OROMANDIBULAR = "Distonia Oromandibular"
    DISARTRIA_FLACIDA = "Disartria Flácida"
    DISARTRIA_ESPASTICA = "Disartria Espástica"
    DISARTRIA_ATAXICA = "Disartria Atáxica"
    NENHUM = "Nenhum"

@dataclass
class PerfilDeFala:
    """
    Representa o perfil clínico do usuário, guiando a correção fonética.
    """
    disturbio_primario: TipoDisturbio = TipoDisturbio.NENHUM
    caracteristicas_observadas: Set[str] = field(default_factory=set)
    intensidade: float = 0.5  # Varia de 0 (leve) a 1 (severo)

    def adicionar_caracteristica(self, desc: str):
        self.caracteristicas_observadas.add(desc)
        logger.info(f"Característica adicionada ao perfil: {desc}")

@dataclass
class EntradaLexico:
    """
    Estrutura de dados para cada mapeamento no léxico.
    """
    transcricao_crua: str
    texto_confirmado: str
    contagem_confirmacao: int = 1
    similaridade_media: float = 1.0
    ultimo_uso: str = field(default_factory=lambda: datetime.now().isoformat())
    fonemas_associados: List[str] = field(default_factory=list)
    contexto_clinico: Optional[str] = None # Armazena o distúrbio ativo durante o aprendizado

    def atualizar_uso(self):
        self.ultimo_uso = datetime.now().isoformat()

class AnalisadorFonetico:
    """
    Módulo de análise fonética que extrai padrões e calcula distâncias.
    """
    def extrair_fonemas(self, texto: str) -> List[str]:
        texto_limpo = re.sub(r'[^a-záàâãéêíóôõúç\s]', '', texto.lower())
        substituicoes = {'ss': 's', 'rr': 'R', 'lh': 'L', 'nh': 'N', 'ch': 'X', 'qu': 'k', 'gu': 'g'}
        for k, v in substituicoes.items():
            texto_limpo = texto_limpo.replace(k, v)
        return list(texto_limpo.replace(' ', ''))

    def calcular_distancia_fonetica(self, palavra1: str, palavra2: str) -> float:
        fonemas1 = self.extrair_fonemas(palavra1)
        fonemas2 = self.extrair_fonemas(palavra2)
        return 1.0 - difflib.SequenceMatcher(None, fonemas1, fonemas2).ratio()

class AnalisadorClinicoFonetico:
    """
    Módulo que traduz conhecimento clínico em regras de correção.
    """
    def __init__(self, perfil: PerfilDeFala):
        self.perfil = perfil
        self.mapa_regras_clinicas = self._mapear_regras()
        logger.info(f"Analisador Clínico inicializado para o perfil: {self.perfil.disturbio_primario.value}")

    def _mapear_regras(self) -> Dict[TipoDisturbio, Dict]:
        return {
            TipoDisturbio.DISTONIA_OROMANDIBULAR: {
                "descricao": "Afeta boca, língua, garganta. Causa fala pastosa/língua pesada.",
                "heuristica": self._heuristica_fala_pastosa
            },
            TipoDisturbio.DISTONIA_LARINGEA: {
                "descricao": "Afeta cordas vocais. Causa fala entrecortada/estrangulada.",
                "heuristica": self._heuristica_fala_entrecortada
            },
            TipoDisturbio.DISTONIA_CERVICAL: {
                "descricao": "Compromete musculatura do pescoço. Causa fala tremida.",
                "heuristica": self._heuristica_fala_tremida
            }
        }

    def aplicar_correcao_preditiva(self, texto: str) -> str:
        if self.perfil.disturbio_primario == TipoDisturbio.NENHUM:
            return texto

        regras = self.mapa_regras_clinicas.get(self.perfil.disturbio_primario)
        if not regras or 'heuristica' not in regras:
            return texto

        logger.info(f"Aplicando heurística para {self.perfil.disturbio_primario.name}")
        texto_corrigido = regras['heuristica'](texto)
        return texto_corrigido

    def _heuristica_fala_pastosa(self, texto: str) -> str:
        correcoes = {
            r'\b(t|d)e\b': 'de',
            r'\b(p|b)ara\b': 'para',
            r'\b(s|z)ua\b': 'sua',
        }
        texto_corrigido = texto
        for padrao, substituicao in correcoes.items():
            texto_corrigido = re.sub(padrao, substituicao, texto_corrigido, flags=re.IGNORECASE)
        return texto_corrigido

    def _heuristica_fala_entrecortada(self, texto: str) -> str:
        texto_corrigido = re.sub(r'(\w)\s+([aeiou]\w*)', r'\1\2', texto)
        return texto_corrigido

    def _heuristica_fala_tremida(self, texto: str) -> str:
        palavras = texto.split()
        palavras_corrigidas = []
        for palavra in palavras:
            if len(palavra) > 4 and palavra[-2:] == palavra[-4:-2]:
                palavras_corrigidas.append(palavra[:-2])
            else:
                palavras_corrigidas.append(palavra)
        return ' '.join(palavras_corrigidas)

class LexicoFoneticoManager:
    """Classe principal para gerenciamento do Léxico Fonético Personalizado."""
    def __init__(self, arquivo_lexico: str, perfil_fala: PerfilDeFala):
        self.arquivo_lexico = Path(arquivo_lexico)
        self.lexico: Dict[str, EntradaLexico] = {}
        self.perfil_fala = perfil_fala
        self.analisador_fonetico = AnalisadorFonetico()
        self.analisador_clinico = AnalisadorClinicoFonetico(self.perfil_fala)

        self._carregar_lexico()
        logger.info(f"Léxico Fonético inicializado com {len(self.lexico)} entradas.")

    def _carregar_lexico(self):
        try:
            if self.arquivo_lexico.exists():
                with open(self.arquivo_lexico, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                for chave, entrada_dict in dados.items():
                    self.lexico[chave] = EntradaLexico(**entrada_dict)
                logger.info(f"Léxico carregado com {len(self.lexico)} entradas.")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Erro ao carregar léxico: {e}. Iniciando com léxico vazio.")
            self.lexico = {}

    def _salvar_lexico(self):
        try:
            with open(self.arquivo_lexico, 'w', encoding='utf-8') as f:
                json.dump({k: asdict(v) for k, v in self.lexico.items()}, f, ensure_ascii=False, indent=2)
            logger.info(f"Léxico salvo com {len(self.lexico)} entradas.")
        except Exception as e:
            logger.error(f"Erro ao salvar léxico: {e}")

    def aprender_mapeamento(self, transcricao_crua: str, texto_confirmado: str):
        chave = self._normalizar_texto(transcricao_crua)
        texto_confirmado_norm = self._normalizar_texto(texto_confirmado)

        if chave in self.lexico:
            entrada = self.lexico[chave]
            if entrada.texto_confirmado == texto_confirmado_norm:
                entrada.contagem_confirmacao += 1
                entrada.atualizar_uso()
                logger.info(f"Reforço de mapeamento: '{chave}' -> '{texto_confirmado_norm}' (x{entrada.contagem_confirmacao})")
            else:
                if entrada.contagem_confirmacao < 2:
                    logger.warning(f"Substituindo mapeamento conflitante para '{chave}'. Novo: '{texto_confirmado_norm}'. Antigo: '{entrada.texto_confirmado}'.")
                    entrada.texto_confirmado = texto_confirmado_norm
                    entrada.contagem_confirmacao = 1
                    entrada.atualizar_uso()
                else:
                     logger.warning(f"Conflito ignorado para '{chave}' devido à alta contagem do mapeamento existente.")
        else:
            logger.info(f"Novo mapeamento aprendido: '{chave}' -> '{texto_confirmado_norm}'")
            nova_entrada = EntradaLexico(
                transcricao_crua=chave,
                texto_confirmado=texto_confirmado_norm,
                contexto_clinico=self.perfil_fala.disturbio_primario.name
            )
            self.lexico[chave] = nova_entrada

        self._salvar_lexico()

    def corrigir_transcricao(self, texto_transcrito: str) -> str:
        texto_normalizado = self._normalizar_texto(texto_transcrito)

        texto_preditivo = self.analisador_clinico.aplicar_correcao_preditiva(texto_normalizado)
        if texto_preditivo != texto_normalizado:
            logger.info(f"Correção Preditiva aplicada: '{texto_normalizado}' -> '{texto_preditivo}'")
            texto_base = texto_preditivo
        else:
            texto_base = texto_normalizado

        if texto_base in self.lexico:
            entrada = self.lexico[texto_base]
            entrada.atualizar_uso()
            self._salvar_lexico()
            logger.info(f"Correção por Léxico: '{texto_base}' -> '{entrada.texto_confirmado}'")
            return entrada.texto_confirmado

        melhor_match = self._buscar_por_similaridade(texto_base)
        if melhor_match:
            entrada, similaridade = melhor_match
            if similaridade > 0.85 and entrada.contagem_confirmacao >= 2:
                entrada.atualizar_uso()
                self._salvar_lexico()
                logger.info(f"Correção por Similaridade: '{texto_base}' -> '{entrada.texto_confirmado}' (sim: {similaridade:.2f})")
                return entrada.texto_confirmado

        if texto_preditivo != texto_normalizado:
            return texto_preditivo
            
        logger.info(f"Nenhuma correção forte encontrada para: '{texto_transcrito}'")
        return texto_transcrito

    def _normalizar_texto(self, texto: str) -> str:
        return re.sub(r'\s+', ' ', texto.lower().strip())

    def _buscar_por_similaridade(self, texto: str) -> Optional[Tuple[EntradaLexico, float]]:
        if not self.lexico: return None
        
        melhor_entrada = None
        maior_similaridade = 0.0

        for chave, entrada in self.lexico.items():
            similaridade = 1 - self.analisador_fonetico.calcular_distancia_fonetica(texto, chave)
            similaridade_ponderada = similaridade * (1 + (entrada.contagem_confirmacao / 10))

            if similaridade_ponderada > maior_similaridade:
                maior_similaridade = similaridade_ponderada
                melhor_entrada = entrada

        if melhor_entrada and maior_similaridade > 0.8: 
            original_sim = 1 - self.analisador_fonetico.calcular_distancia_fonetica(texto, melhor_entrada.transcricao_crua)
            return melhor_entrada, original_sim
            
        return None

# --- Processador de Fala Adaptativo (do fala_adaptativa.py) ---
@dataclass
class UserProfile:
    """Perfil personalizado do usuário"""
    user_id: str
    severity_level: int  # 1-5
    dominant_symptoms: List[str]
    preferred_corrections: Dict[str, float]
    adaptation_history: List[Dict]
    biometric_baselines: Dict[str, float]

class AdaptiveSpeechProcessor:
    """
    Sistema de processamento de fala com distonia
    Integra modelos transformer, difusão, e IA multimodal
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 device: str = "auto"):
        
        self.sample_rate = sample_rate
        self.device = self._setup_device(device)
        
        # Inicializa componentes avançados
        self._initialize_advanced_models()
        
        # Sistema de processamento distribuído
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("✅ Sistema Avançado de Processamento inicializado")
    
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _initialize_advanced_models(self):
        """Inicializa modelos de deep learning avançados"""
        if not TORCH_AVAILABLE or not HAS_SPEECH_ANALYSIS_LIBS:
            logger.warning("Modelos avançados de fala não disponíveis devido a dependências ausentes.")
            return
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration
            # Whisper para transcrição de alta qualidade
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            
            # Wav2Vec2 para análise prosódica
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
            
            logger.info("✅ Modelos avançados carregados")
        except Exception as e:
            logger.warning(f"⚠️ Alguns modelos não puderam ser carregados: {e}")

    async def analyze_audio_for_adaptation(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Realiza análise de áudio para adaptação de fala."""
        if not HAS_SPEECH_ANALYSIS_LIBS:
            return {"error": "Bibliotecas de análise de fala não disponíveis."}
        
        analysis_results = {}
        try:
            # Exemplo: extração de pitch e intensidade
            # Usando parselmouth para análise de pitch
            from parselmouth.praat import call as praat_call
            sound = parselmouth.Sound(audio_data, self.sample_rate)
            pitch = praat_call(sound, "To Pitch", 0.0, 75, 600)  # min/max pitch
            intensity = praat_call(sound, "To Intensity", 75, 600) # min/max intensity

            analysis_results['pitch_contour'] = pitch.get_values_at_times(pitch.get_time_domain()).tolist()
            analysis_results['intensity_contour'] = intensity.get_values_at_times(intensity.get_time_domain()).tolist()
            analysis_results['avg_pitch'] = praat_call(pitch, "Get mean", 0, 0, "Hertz")
            analysis_results['avg_intensity'] = praat_call(intensity, "Get mean", 0, 0, "dB")

            # Exemplo: transcrição com Whisper (se disponível)
            if hasattr(self, 'whisper_model') and self.whisper_model:
                input_features = self.whisper_processor(audio_data, sampling_rate=self.sample_rate, return_tensors="pt").input_features
                predicted_ids = self.whisper_model.generate(input_features.to(self.device))
                transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                analysis_results['transcription'] = transcription

        except Exception as e:
            logger.error(f"Erro na análise de áudio para adaptação: {e}")
            analysis_results['error'] = str(e)

        return analysis_results

# --- Orquestrador Principal do Motor de Voz ---
class AtenaVoiceMotor:
    """
    Motor de Voz unificado da Atena.
    Orquestra a síntese de fala, análise adaptativa e correção fonética.
    """
    def __init__(self, cache_dir: str = "./atena_tts_cache", 
                 lexico_path: str = "./memoria_do_usuario/lexico_fonetico.json",
                 user_profile: Optional[UserProfile] = None):
        
        self.tts_engine = UnifiedTTSEngine(cache_dir=cache_dir)
        self.adaptive_processor = AdaptiveSpeechProcessor() # Para análise de fala
        
        # Perfil de fala padrão se não for fornecido
        self.user_profile = user_profile or UserProfile(
            user_id="default_user", 
            severity_level=1, 
            dominant_symptoms=[], 
            preferred_corrections={}, 
            adaptation_history=[], 
            biometric_baselines={}
        )
        self.lexico_manager = LexicoFoneticoManager(lexico_path, self.user_profile)
        
        logger.info("🎤 Atena Voice Motor inicializado.")

    async def speak(self, text: str, persona: str = "normal", play_audio: bool = True) -> Tuple[bytes, AudioMetrics]:
        """
        Sintetiza e reproduz fala, aplicando adaptações e correções.
        """
        # 1. Correção Fonética e Adaptação Clínica (antes da síntese)
        corrected_text = self.lexico_manager.corrigir_transcricao(text)
        
        # 2. Análise de Sentimento e Prosódia (via NLPProcessor do TTS Engine)
        # O NLPProcessor já está integrado no synthesize do UnifiedTTSEngine
        
        # 3. Síntese de Fala
        audio_data, metrics = await self.tts_engine.synthesize_async(corrected_text, apply_nlp=True) # apply_nlp para prosody adjustments
        
        # 4. Reprodução
        if play_audio:
            self.tts_engine.play_audio(audio_data)
            
        return audio_data, metrics

    async def analyze_user_speech(self, audio_input: bytes) -> Dict[str, Any]:
        """
        Analisa a fala do usuário para feedback e adaptação do perfil.
        """
        # Converte bytes para numpy array para análise
        audio_array = np.frombuffer(audio_input, dtype=np.int16)
        
        analysis_results = await self.adaptive_processor.analyze_audio_for_adaptation(audio_array)
        
        # Aqui, você pode usar analysis_results para atualizar o user_profile
        # Por exemplo, se detectar padrões de fala específicos de um distúrbio
        # self.user_profile.adicionar_caracteristica("detectado_fala_pastosa")
        
        return analysis_results

    def get_status(self) -> Dict[str, Any]:
        tts_stats = self.tts_engine.get_system_stats()
        lexico_stats = self.lexico_manager.get_memory_stats()
        
        return {
            "tts_engine_status": tts_stats,
            "lexico_manager_status": lexico_stats,
            "user_profile_summary": asdict(self.user_profile),
            "overall_health": "healthy" # Simplificado
        }

    def shutdown(self):
        self.tts_engine.shutdown()
        logger.info("🎤 Atena Voice Motor finalizado.")

# --- Exemplo de Uso e Demonstração ---
async def main_demo():
    print("=== DEMO: Atena Voice Motor Unificado ===\n")

    # Configurar um perfil de fala de exemplo
    perfil_exemplo = PerfilDeFala(
        user_id="roberio_user",
        disturbio_primario=TipoDisturbio.DISTONIA_OROMANDIBULAR,
        intensidade=0.7,
        dominant_symptoms=["fala pastosa", "dificuldade em consoantes"],
        preferred_corrections={'speed': 1.1}
    )

    # Inicializar o motor de voz
    voice_motor = AtenaVoiceMotor(user_profile=perfil_exemplo)

    # Testar síntese de fala com correção e adaptação
    phrases_to_synthesize = [
        "Olá, eu sou a Atena, sua assistente virtual.",
        "A grama é verde e o céu é azul.",
        "Eu preciso de ajuda com a minha fala.",
        "atena ligar lus", # Exemplo de erro de transcrição para correção
        "bodia atena", # Exemplo de erro de transcrição para correção
    ]

    for i, phrase in enumerate(phrases_to_synthesize):
        print(f"\nProcessando frase {i+1}: '{phrase}'")
        audio_data, metrics = await voice_motor.speak(phrase, persona="normal", play_audio=True)
        print(f"  -> Áudio gerado ({len(audio_data)} bytes), Qualidade: {metrics.quality_score:.2f}")
        await asyncio.sleep(2) # Pequena pausa para audição

    # Simular aprendizado de mapeamento fonético
    print("\n--- Simulando Aprendizado Fonético ---")
    voice_motor.lexico_manager.aprender_mapeamento("atena ligar lus", "atena ligar luz")
    voice_motor.lexico_manager.aprender_mapeamento("bodia atena", "bom dia atena")

    # Testar novamente a frase corrigida
    print("\nProcessando frase corrigida: 'atena ligar lus' (após aprendizado)")
    audio_data, metrics = await voice_motor.speak("atena ligar lus", persona="normal", play_audio=True)
    print(f"  -> Áudio gerado ({len(audio_data)} bytes), Qualidade: {metrics.quality_score:.2f}")
    await asyncio.sleep(2)

    # Obter e exibir status
    print("\n--- Status do Motor de Voz ---")
    status = voice_motor.get_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))

    voice_motor.shutdown()
    print("\nDemo concluída!")

if __name__ == "__main__":
    # Para executar a demonstração, certifique-se de ter as dependências instaladas:
    # pip install edge-tts pydub pygame numpy nltk transformers librosa praat-parselmouth
    # E baixar os modelos NLTK: nltk.download('punkt'), nltk.download('stopwords'), nltk.download('vader_lexicon')
    asyncio.run(main_demo())
