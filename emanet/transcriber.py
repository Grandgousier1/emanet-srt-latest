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
    Handles audio transcription using an ASR model.
    Currently uses faster-whisper, but designed to be adaptable.
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
    ):
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper n'est pas installé. "
                "Veuillez l'installer avec : pip install faster-whisper"
            )

        # NOTE: This section is designed for faster-whisper.
        # To implement Voxtral, a new loading mechanism will be needed here.
        # The 'transcribe' method signature should be kept compatible.
        logger.info(f"Chargement du modèle de transcription ASR: {model_name}...")
        logger.info("NOTE : Le modèle Voxtral n'étant pas disponible, "
                    "nous utilisons faster-whisper comme solution de repli.")

        self.model_name = model_name
        self.device = device
        self.model = WhisperModel(
            model_name, device=device, compute_type=compute_type, download_root=model_dir
        )

        self.use_whisperx_align = use_whisperx_align and HAS_WHISPERX
        if use_whisperx_align and not HAS_WHISPERX:
            logger.warning("WhisperX n'est pas installé, l'alignement est désactivé. "
                           "Installez-le avec pip si vous le souhaitez.")

        self.beam_size = beam_size
        self.best_of = best_of

    def transcribe(self, audio_path: str, language: str = "tr") -> List[Segment]:
        logger.info(f"Transcription de {os.path.basename(audio_path)}...")
        t0 = time.time()

        # The transcription process with faster-whisper returns an iterator.
        # We wrap it with tqdm for a progress bar.
        it, info = self.model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=0.0,
            patience=0.0,
            word_timestamps=True,
            condition_on_previous_text=True,
        )

        segments = []
        total_duration = round(info.duration)

        with tqdm(total=total_duration, unit="s", desc="Transcription ASR") as pbar:
            for seg in it:
                if seg.words:
                    words = [Word(w.word, w.start, w.end) for w in seg.words if w.word.strip()]
                else:
                    words = None

                segments.append(
                    Segment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text.strip(),
                        words=words,
                        avg_logprob=getattr(seg, "avg_logprob", None),
                        no_speech_prob=getattr(seg, "no_speech_prob", None),
                    )
                )
                pbar.update(round(seg.end - pbar.n))

        logger.info(f"Transcription terminée en {time.time() - t0:.1f}s — {len(segments)} segments trouvés.")

        # Filter segments with low confidence
        segments = self._filter_segments(segments)

        if self.use_whisperx_align:
            try:
                segments = self._align_with_whisperx(audio_path, segments, language)
            except Exception as e:
                logger.warning(f"L'alignement WhisperX a échoué ({e}) — on continue sans.")

        return segments

    def _filter_segments(self, segments: List[Segment]) -> List[Segment]:
        filtered = []
        for s in segments:
            if s.no_speech_prob is not None and s.no_speech_prob > 0.6:
                continue
            if s.avg_logprob is not None and s.avg_logprob < -1.25:
                continue
            filtered.append(s)
        if len(segments) != len(filtered):
            logger.info(f"{len(segments) - len(filtered)} segments de faible confiance ont été filtrés.")
        return filtered

    def _align_with_whisperx(self, audio_path: str, segments: List[Segment], language: str) -> List[Segment]:
        if not HAS_WHISPERX:
            return segments

        logger.info("Alignement mot-à-mot avec WhisperX...")
        device = self.device if torch.cuda.is_available() else "cpu"

        try:
            align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        except Exception as e:
            logger.error(f"Impossible de charger le modèle d'alignement pour la langue '{language}'. Erreur: {e}")
            return segments

        wx_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]

        result = whisperx.align(
            transcript={"segments": wx_segments},
            align_model=align_model,
            audio=audio_path,
            device=device,
        )

        aligned = []
        for s, wx in zip(segments, result["segments"]):
            words = []
            if "words" in wx:
                for w in wx["words"]:
                    if "start" in w and "end" in w and "word" in w:
                        if w["start"] is None or w["end"] is None:
                            continue
                        words.append(Word(w["word"], float(w["start"]), float(w["end"])))

            # If alignment fails for a segment, fall back to original words if available
            if not words and s.words:
                words = s.words

            aligned.append(Segment(start=wx["start"], end=wx["end"], text=wx["text"].strip(), words=words))

        logger.info(f"Alignement WhisperX terminé — {len(aligned)} segments alignés.")
        return aligned
