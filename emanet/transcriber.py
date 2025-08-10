import os
import time
import logging
from typing import List, Optional

import torch
import torchaudio
from tqdm import tqdm

from .datastructures import Word, Segment
from .utils import SileroVAD

logger = logging.getLogger("emanet")

# Tentative d'importation des bibliothèques liées à la transcription
try:
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    import bitsandbytes
    HAS_VOXTRAL_DEPS = True
except ImportError:
    HAS_VOXTRAL_DEPS = False


class Transcriber:
    """
    Gère la transcription audio en utilisant le modèle Voxtral de Mistral.
    Utilise SileroVAD pour la détection d'activité vocale en amont.
    """
    def __init__(
        self,
        model_name: str = "mistralai/Voxtral-Mini-3B-2507",
        device: str = "cpu",
        quant: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        """
        Initialise le transcriber avec Voxtral.

        Args:
            model_name (str): Nom du modèle Voxtral à utiliser depuis Hugging Face.
            device (str): "cuda" ou "cpu".
            quant (Optional[str]): Type de quantification (ex: "int4", "int8").
                                   Nécessite un GPU compatible.
            model_dir (Optional[str]): Dossier pour télécharger les modèles.
        """
        if not HAS_VOXTRAL_DEPS:
            raise RuntimeError("Dépendances Voxtral non installées. Installez-les avec : pip install transformers bitsandbytes")

        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialisation du transcriber sur le device : {self.device}")

        quant_config = None
        if self.device == "cuda":
            if quant == "int8":
                quant_config = bitsandbytes.BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Quantification 8-bit activée.")
            elif quant == "int4":
                quant_config = bitsandbytes.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                logger.info("Quantification 4-bit activée.")
        elif quant:
            logger.warning(f"La quantification '{quant}' n'est supportée que sur GPU, elle est ignorée sur CPU.")

        logger.info(f"Chargement du modèle de transcription Voxtral : {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_dir)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            quantization_config=quant_config,
            cache_dir=model_dir,
            low_cpu_mem_usage=True if self.device == "cuda" else False,
        )
        if quant_config is None:
            self.model.to(self.device)

        logger.info("Modèle Voxtral chargé.")

    def transcribe(self, audio_path: str, language: str) -> List[Segment]:
        """
        Transcrire un fichier audio en utilisant VAD + Voxtral.

        Args:
            audio_path (str): Chemin vers le fichier audio.
            language (str): Code de la langue de l'audio (ex: "fr", "en").

        Returns:
            List[Segment]: Une liste de segments transcrits.
        """
        logger.info(f"Lancement de la VAD pour {os.path.basename(audio_path)}...")
        t0 = time.time()

        speech_timestamps = SileroVAD.get_speech_timestamps(audio_path)
        if not speech_timestamps:
            logger.warning("Aucun segment de parole détecté par la VAD.")
            return []

        logger.info(f"Transcription de {len(speech_timestamps)} segments de parole...")

        full_audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            full_audio = resampler(full_audio)
            sample_rate = 16000

        all_segments = []

        with tqdm(total=len(speech_timestamps), unit="segment", desc="Transcription Voxtral") as pbar:
            for chunk in speech_timestamps:
                start_time, end_time = chunk['start'], chunk['end']

                # Extrait le segment audio
                audio_chunk = full_audio[:, int(start_time * sample_rate):int(end_time * sample_rate)]

                inputs = self.processor.apply_transcription_request(
                    audio=audio_chunk.squeeze(0),
                    sampling_rate=sample_rate,
                    language=language,
                    model_id=self.model.config.name_or_path,
                    return_timestamps="word"
                )

                inputs = inputs.to(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32)

                with torch.no_grad():
                    result = self.model.generate(**inputs, max_new_tokens=500)

                decoded = self.processor.batch_decode(
                    result.sequences,
                    skip_special_tokens=True,
                    output_word_offsets=True
                )

                if decoded and decoded[0]:
                    transcription = decoded[0][0]
                    text = transcription["text"]
                    word_offsets = transcription.get("word_offsets", [])

                    words = [Word(w['word'], w['start_offset'], w['end_offset']) for w in word_offsets]

                    # Ajuste les timestamps des mots pour être relatifs à l'audio complet
                    for word in words:
                        word.start += start_time
                        word.end += start_time

                    segment = Segment(
                        start=start_time,
                        end=end_time,
                        text=text.strip(),
                        words=words
                    )
                    all_segments.append(segment)

                pbar.update(1)

        logger.info(f"Transcription terminée en {time.time() - t0:.1f}s — {len(all_segments)} segments trouvés.")
        return all_segments
