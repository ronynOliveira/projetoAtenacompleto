#!/usr/bin/env python3
"""
Atena Voice Engine v2.0 - Sistema de Síntese de Fala Híbrido e Adaptativo
Aprimorado com motor TTS local, prosódia de alta qualidade e cache otimizado.

Autor: Claude & Gemini (para Atena e Senhor Robério)
Versão: 2.0
"""

import os
import hashlib
import json
import re
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

# --- Dependências de Áudio e TTS ---
# Tenta importar as bibliotecas e define flags de disponibilidade

try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    print("Warning: pydub não encontrado. Funções de manipulação e reprodução de áudio estarão desabilitadas. Instale com: pip install pydub")
    PYDUB_AVAILABLE = False
    AudioSegment = None  # Define um placeholder

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    print("Warning: pyttsx3 não encontrado. A geração de fala local estará desabilitada. Instale com: pip install pyttsx3")
    PYTTSX3_AVAILABLE = False

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AtenaVoiceEngineV2')


class SmartCacheManager:
    """
    Gerenciador inteligente de cache com dois níveis e persistência.
    """
    def __init__(self, base_cache_dir: str = "./atena_tts_cache"):
        self.base_dir = Path(base_cache_dir)
        self.phrases_dir = self.base_dir / "phrases"
        self.words_dir = self.base_dir / "words"
        self.metadata_dir = self.base_dir / "metadata"
        self.stats = defaultdict(int)
        self._setup_cache_structure()

    def _setup_cache_structure(self):
        for dir_path in [self.phrases_dir, self.words_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.phrase_index_file = self.metadata_dir / "phrase_index.json"
        self.word_index_file = self.metadata_dir / "word_index.json"
        self.phrase_index = self._load_index(self.phrase_index_file)
        self.word_index = self._load_index(self.word_index_file)
        logger.info(f"Cache inicializado em: {self.base_dir}")

    def _load_index(self, index_file: Path) -> Dict:
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Erro ao carregar índice {index_file}: {e}. Criando um novo.")
        return {}

    def _save_index(self, index: Dict, index_file: Path):
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Erro ao salvar índice {index_file}: {e}")

    def _generate_key(self, text: str, voice_params: Dict = None) -> str:
        """Gera um hash único para o texto e os parâmetros de voz."""
        normalized_text = re.sub(r'[^\w\s]', '', text.lower().strip())
        normalized_text = re.sub(r'\s+', ' ', normalized_text)
        
        # Inclui parâmetros de voz no hash para cache de prosódia
        params_str = ""
        if voice_params:
            params_str = json.dumps(sorted(voice_params.items()))
        
        key_data = f"{normalized_text}|{params_str}".encode('utf-8')
        return hashlib.sha256(key_data).hexdigest()

    def get_phrase(self, text: str, voice_params: Dict = None) -> Optional[AudioSegment]:
        if not PYDUB_AVAILABLE: return None
        phrase_hash = self._generate_key(text, voice_params)
        file_info = self.phrase_index.get(phrase_hash)
        
        if file_info:
            file_path = self.phrases_dir / file_info['file']
            if file_path.exists():
                try:
                    audio = AudioSegment.from_wav(str(file_path))
                    self.stats['cache_hits_phrases'] += 1
                    logger.debug(f"Cache HIT (frase): {text[:50]}...")
                    return audio
                except Exception as e:
                    logger.error(f"Erro ao carregar áudio da frase do cache: {e}")
        return None

    def save_phrase(self, text: str, audio_data: AudioSegment, voice_params: Dict = None, metadata: Dict = None):
        if not PYDUB_AVAILABLE: return
        phrase_hash = self._generate_key(text, voice_params)
        filename = f"{phrase_hash}.wav"
        file_path = self.phrases_dir / filename
        
        try:
            audio_data.export(str(file_path), format="wav")
            self.phrase_index[phrase_hash] = {
                'text': text, 'file': filename, 'created_at': time.time(),
                'voice_params': voice_params or {}, 'metadata': metadata or {}
            }
            self._save_index(self.phrase_index, self.phrase_index_file)
            self.stats['phrases_generated'] += 1
            logger.debug(f"Frase salva no cache: {text[:50]}...")
        except Exception as e:
            logger.error(f"Erro ao salvar frase no cache: {e}")

    def get_word(self, word: str) -> Optional[AudioSegment]:
        if not PYDUB_AVAILABLE: return None
        normalized_word = self._normalize_word(word)
        file_info = self.word_index.get(normalized_word)

        if file_info:
            file_path = self.words_dir / file_info['file']
            if file_path.exists():
                try:
                    audio = AudioSegment.from_wav(str(file_path))
                    self.stats['cache_hits_words'] += 1
                    return audio
                except Exception as e:
                    logger.error(f"Erro ao carregar áudio da palavra '{word}': {e}")
        return None

    def save_word(self, word: str, audio_data: AudioSegment):
        if not PYDUB_AVAILABLE: return
        normalized_word = self._normalize_word(word)
        filename = f"{normalized_word}.wav"
        file_path = self.words_dir / filename

        try:
            audio_data.export(str(file_path), format="wav")
            self.word_index[normalized_word] = {
                'original_word': word, 'file': filename, 'created_at': time.time()
            }
            self._save_index(self.word_index, self.word_index_file)
            self.stats['words_learned'] += 1
        except Exception as e:
            logger.error(f"Erro ao salvar palavra no léxico: {e}")

    def _normalize_word(self, word: str) -> str:
        return re.sub(r'[^\w]', '', word.lower().strip())
        
    def get_stats(self) -> Dict:
        self.stats['total_phrases_cached'] = len(self.phrase_index)
        self.stats['total_words_cached'] = len(self.word_index)
        total_hits = self.stats['cache_hits_phrases'] + self.stats['cache_hits_words']
        total_requests = total_hits + self.stats['cache_misses']
        self.stats['cache_efficiency'] = total_hits / max(1, total_requests)
        return self.stats

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

class LocalTTSProvider:
    """Provedor de TTS local usando pyttsx3."""
    def __init__(self):
        if not PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 não é encontrado. O motor de TTS local não pode ser iniciado.")
        self.engine = pyttsx3.init()
        self.temp_file_path = Path("./tts_temp.wav")
        logger.info(f"TTS Provider local (pyttsx3) inicializado. Voz padrão: {self.engine.getProperty('voice')}")

    def synthesize(self, text: str, voice_params: Dict = None) -> Optional[AudioSegment]:
        if not PYDUB_AVAILABLE: return None
        try:
            # Aplicar parâmetros de voz antes da síntese
            if voice_params:
                if 'velocidade' in voice_params:
                    # O rate padrão é 200. Um fator de 1.0 = 200.
                    self.engine.setProperty('rate', 200 * voice_params['velocidade'])
                if 'volume' in voice_params:
                    self.engine.setProperty('volume', voice_params['volume'])
                # Nota: pyttsx3 não tem um controle de 'tom' (pitch) tão direto e confiável.
                # A manipulação de pitch é mais complexa e dependente do SAPI/driver.

            self.engine.save_to_file(text, str(self.temp_file_path))
            self.engine.runAndWait()
            
            if self.temp_file_path.exists():
                audio = AudioSegment.from_wav(str(self.temp_file_path))
                os.remove(self.temp_file_path) # Limpa o arquivo temporário
                return audio
            else:
                logger.error("Falha na geração do arquivo de áudio temporário pelo pyttsx3.")
                return None
        except Exception as e:
            logger.error(f"Erro na síntese com pyttsx3: {e}", exc_info=True)
            return None

class AtenaVoiceEngine:
    """
    Motor de síntese de fala v2.0.
    Implementa um sistema híbrido de cache e geração local.
    """
    def __init__(self, cache_dir: str = "./atena_tts_cache"):
        self.cache = SmartCacheManager(cache_dir)
        self.processor = AudioProcessor()
        self.local_tts = LocalTTSProvider() if PYTTSX3_AVAILABLE else None
        
        self.personas = {
            'normal':       {'velocidade': 1.0, 'volume': 1.0},
            'entusiasmada': {'velocidade': 1.15, 'volume': 1.0},
            'pensativa':    {'velocidade': 0.85, 'volume': 0.9},
            'confiante':    {'velocidade': 1.05, 'volume': 1.0},
        }
        self.total_speeches = 0
        self.stage_usage = defaultdict(int)
        logger.info("🎙️ Atena Voice Engine v2.0 inicializado com sucesso!")

    def speak(self, text: str, persona: str = 'normal', play: bool = True) -> Optional[AudioSegment]:
        if not text.strip():
            logger.warning("Texto vazio fornecido para síntese.")
            return None
        if not self.local_tts:
            logger.error("Motor TTS local (pyttsx3) não está disponível. Não é possível gerar fala.")
            return None
            
        self.total_speeches += 1
        start_time = time.time()
        logger.info(f"🗣️ Atena fala: '{text[:100]}...' (Persona: {persona})")
        
        voice_params = self.personas.get(persona, self.personas['normal'])

        # Estágio 1: Busca no cache de frases (com prosódia)
        final_audio = self.cache.get_phrase(text, voice_params)
        if final_audio:
            self.stage_usage[1] += 1
            self._log_performance(start_time, "Cache Nível 1 (Frase com Prosódia)")
            if play:
                self.processor.play_audio_async(final_audio)
            return final_audio
        
        # Se não encontrou a frase com prosódia, o processo continua para compor ou gerar.
        self.cache.stats['cache_misses'] += 1
        words = self._extract_words(text)
        
        # Estágio 2: Composição dinâmica a partir do léxico de palavras
        logger.debug("Tentando composição a partir do léxico...")
        
        word_audios = []
        words_to_generate = []
        for word in words:
            word_audio = self.cache.get_word(word)
            if word_audio:
                word_audios.append(word_audio)
            else:
                # Se uma palavra falta, não podemos compor. Precisamos gerar.
                logger.debug(f"Cache MISS (palavra): '{word}'. Geração online necessária.")
                words_to_generate.append(word)
        
        # Se alguma palavra faltou, vamos para o estágio de geração
        if words_to_generate:
            self.stage_usage[3] += 1
            logger.info(f"🌐 Gerando {len(words_to_generate)} palavras faltantes...")
            
            # Gerar apenas as palavras que faltam
            for missing_word in set(words_to_generate):
                # O motor pyttsx3 não tem prosódia por palavra, então usamos a padrão
                word_audio_gen = self.local_tts.synthesize(missing_word)
                if word_audio_gen:
                    self.cache.save_word(missing_word, word_audio_gen)
            
            # Tentar compor novamente com o cache agora enriquecido
            word_audios = [self.cache.get_word(w) for w in words]
            word_audios = [audio for audio in word_audios if audio is not None]

        else: # Todas as palavras estavam no cache
            self.stage_usage[2] += 1
            logger.info("✅ Composição do léxico bem-sucedida!")

        # Se temos todos os áudios de palavras, concatenamos
        if len(word_audios) == len(words):
            composed_audio = self.processor.concatenate_words(word_audios)
            # Gerar a frase completa com a prosódia desejada
            full_phrase_audio = self.local_tts.synthesize(text, voice_params)
            
            # Salvar a frase com prosódia no cache de Nível 1
            if full_phrase_audio:
                self.cache.save_phrase(text, full_phrase_audio, voice_params)
                final_audio = full_phrase_audio
                self._log_performance(start_time, "Geração e Cache")
            else: # Fallback para o áudio composto sem prosódia
                final_audio = composed_audio
                self._log_performance(start_time, "Composição do Léxico")

        else:
            logger.error("❌ Falha na geração/composição. Algumas palavras não puderam ser sintetizadas.")
            return None

        if play:
            self.processor.play_audio_async(final_audio)
            
        return final_audio

    def _extract_words(self, text: str) -> List[str]:
        return [word for word in re.findall(r'\b\w+\b', text.lower()) if word]

    def _log_performance(self, start_time: float, stage: str):
        elapsed = time.time() - start_time
        logger.info(f"⚡ Síntese concluída em {elapsed:.3f}s via {stage}")

def demo_atena_voice():
    """Demonstração do Atena Voice Engine v2.0"""
    print("🎭 === DEMO: Atena Voice Engine v2.0 ===")
    
    # Inicializar engine
    atena = AtenaVoiceEngine()
    
    test_phrases = [
        "Olá, eu sou a Atena, sua assistente inteligente.",
        "Como posso ajudá-lo hoje?",
        "Estou aprendendo continuamente para melhor servi-lo.",
        "Minha voz evolui a cada conversa.",
        "Olá, eu sou a Atena, sua assistente inteligente."
    ]
    personas = ['normal', 'entusiasmada', 'pensativa', 'confiante']
    
    print("\n🎤 Sintetizando frases com diferentes personas (a reprodução é assíncrona)...")
    
    for i, phrase in enumerate(test_phrases):
        persona = personas[i % len(personas)]
        print(f"\n[{i+1}] Persona: {persona} | Frase: '{phrase}'")
        atena.speak(phrase, persona=persona, play=True)
        time.sleep(2) # Pausa para não sobrepor os áudios na demonstração

    # Pausa final para o último áudio terminar
    time.sleep(5)
    
    print("\n📊 === ESTATÍSTICAS DO SISTEMA ===")
    stats = atena.cache.get_stats()
    print(json.dumps(stats, indent=2))
    print("\n✅ Demo concluída!")

if __name__ == "__main__":
    if not PYTTSX3_AVAILABLE or not PYDUB_AVAILABLE:
        logger.error("Uma ou mais dependências (pyttsx3, pydub) não estão instaladas. A demonstração não pode continuar.")
    else:
        demo_atena_voice()