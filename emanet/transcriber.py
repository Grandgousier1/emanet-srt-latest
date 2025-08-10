import os
import time
import logging
from typing import List, Optional

import torch
from tqdm import tqdm

from .datastructures import Word, Segment

logger = logging.getLogger("emanet")

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import whisperx
    HAS_WHISPERX = True
except ImportError:
    HAS_WHISPERX = False


class Transcriber:
    """
    Gère la transcription audio en utilisant un modèle ASR (Automatic Speech Recognition).
    Implémentation actuelle utilise faster-whisper, mais est conçue pour être adaptable.
    """
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        model_dir: Optional[str] = None,
        use_whisperx_align: bool = True,
        beam_size: int = 5,
        best_of: int = 5,
        no_speech_threshold: float = 0.6,
        logprob_threshold: float = -1.25,
    ):
        """
        Initialise le transcriber.

        Args:
            model_name (str): Nom du modèle faster-whisper à utiliser.
            device (str): "cuda" ou "cpu".
            compute_type (str): Type de calcul (ex: "float16", "int8_float16").
            model_dir (Optional[str]): Dossier pour télécharger les modèles.
            use_whisperx_align (bool): Activer/désactiver l'alignement avec WhisperX.
            beam_size (int): Taille du faisceau pour le décodage.
            best_of (int): Nombre de candidats à considérer.
            no_speech_threshold (float): Seuil de probabilité pour filtrer les segments sans parole.
            logprob_threshold (float): Seuil de log-probabilité pour filtrer les segments de faible confiance.
        """
        if WhisperModel is None:
            raise RuntimeError("faster-whisper n'est pas installé. Veuillez l'installer avec : pip install faster-whisper")

        logger.info(f"Chargement du modèle de transcription ASR: {model_name} sur {device} ({compute_type})...")
        # NOTE: Cette section est spécifique à faster-whisper. Pour implémenter Voxtral,
        # un nouveau mécanisme de chargement sera nécessaire ici.
        logger.info("NOTE : Le modèle Voxtral n'étant pas trouvable, nous utilisons faster-whisper comme solution de repli.")

        self.model_name = model_name
        self.device = device
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=model_dir)

        self.use_whisperx_align = use_whisperx_align and HAS_WHISPERX
        if use_whisperx_align and not HAS_WHISPERX:
            logger.warning("WhisperX n'est pas installé, l'alignement est désactivé. Installez-le avec 'pip install whisperx' si besoin.")

        self.beam_size = beam_size
        self.best_of = best_of
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold

    def transcribe(self, audio_path: str, language: str = "tr") -> List[Segment]:
        """
        Transcrire un fichier audio.

        Args:
            audio_path (str): Chemin vers le fichier audio.
            language (str): Code de la langue de l'audio (ex: "tr", "en").

        Returns:
            List[Segment]: Une liste de segments transcrits.
        """
        logger.info(f"Transcription de {os.path.basename(audio_path)}...")
        t0 = time.time()

        it, info = self.model.transcribe(
            audio_path, language=language, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=self.beam_size, best_of=self.best_of,
            temperature=0.0, patience=0.0, word_timestamps=True,
            condition_on_previous_text=True,
        )

        segments = []
        total_duration = round(info.duration)

        with tqdm(total=total_duration, unit="s", desc="Transcription ASR") as pbar:
            for seg in it:
                words = [Word(w.word, w.start, w.end) for w in seg.words if w.word.strip()] if seg.words else None
                segments.append(Segment(
                    start=seg.start, end=seg.end, text=seg.text.strip(),
                    words=words, avg_logprob=getattr(seg, "avg_logprob", None),
                    no_speech_prob=getattr(seg, "no_speech_prob", None),
                ))
                pbar.update(round(seg.end - pbar.n))

        logger.info(f"Transcription terminée en {time.time() - t0:.1f}s — {len(segments)} segments trouvés.")

        segments = self._filter_segments(segments)

        if self.use_whisperx_align:
            try:
                segments = self._align_with_whisperx(audio_path, segments, language)
            except Exception as e:
                logger.warning(f"L'alignement WhisperX a échoué ({e}) — on continue sans.")

        return segments

    def _filter_segments(self, segments: List[Segment]) -> List[Segment]:
        """Filtre les segments transcrits en fonction des seuils de confiance."""
        initial_count = len(segments)
        filtered = [
            s for s in segments
            if (s.no_speech_prob is None or s.no_speech_prob <= self.no_speech_threshold)
            and (s.avg_logprob is None or s.avg_logprob >= self.logprob_threshold)
        ]
        if initial_count != len(filtered):
            logger.info(f"{initial_count - len(filtered)} segments de faible confiance ont été filtrés.")
        return filtered

    def _align_with_whisperx(self, audio_path: str, segments: List[Segment], language: str) -> List[Segment]:
        """Aligne les segments transcrits au niveau du mot en utilisant WhisperX."""
        if not HAS_WHISPERX:
            return segments

        logger.info("Alignement mot-à-mot avec WhisperX...")
        device = self.device if torch.cuda.is_available() else "cpu"

        try:
            align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        except Exception as e:
            logger.error(f"Impossible de charger le modèle d'alignement WhisperX pour la langue '{language}'. Erreur: {e}")
            return segments

        wx_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]

        try:
            result = whisperx.align(
                transcript={"segments": wx_segments}, align_model=align_model,
                audio=audio_path, device=device,
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'alignement WhisperX : {e}")
            return segments

        aligned = []
        for s, wx in zip(segments, result["segments"]):
            words = []
            if "words" in wx:
                for w in wx.get("words", []):
                    if "start" in w and "end" in w and "word" in w and w["start"] is not None:
                        words.append(Word(w["word"], float(w["start"]), float(w["end"])))

            # Si l'alignement a échoué pour ce segment, on garde les mots originaux de Whisper
            if not words and s.words:
                words = s.words

            aligned.append(Segment(start=wx["start"], end=wx["end"], text=wx["text"].strip(), words=words))

        logger.info(f"Alignement WhisperX terminé — {len(aligned)} segments alignés.")
        return aligned
