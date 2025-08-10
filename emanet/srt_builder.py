import re
import srt
import logging
from typing import List

from .datastructures import Segment, Cue, Word
from .utils import hhmmssmmm, normalize_whitespace, french_typography_fix, split_balanced_two_lines

logger = logging.getLogger("emanet")

class SRTBuilder:
    """
    Builds a clean, well-formatted SRT file from transcribed segments and translated texts.
    """
    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        max_cps: float = 17.0,
        min_duration: float = 1.0,
        max_duration: float = 7.0,
        gap: float = 0.08,
    ):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.max_cps = max_cps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.gap = gap

    def build_from_segments(self, segments: List[Segment], translated_texts: List[str]) -> List[Cue]:
        # 1. Get all source words with timestamps for timing reference
        source_words = []
        for s in segments:
            if s.words:
                for w in s.words:
                    source_words.append(Word(text=w.text, start=w.start, end=w.end))
            elif s.text.strip(): # Fallback to segment timing if no words
                source_words.append(Word(text=s.text, start=s.start, end=s.end))
        source_words.sort(key=lambda w: (w.start, w.end))

        if not source_words:
            logger.warning("Aucun mot source avec horodatage trouvé. Impossible de construire le SRT.")
            return []

        # 2. Concatenate and tokenize the translated French text
        full_translated_text = " ".join([normalize_whitespace(t) for t in translated_texts])
        full_translated_text = french_typography_fix(full_translated_text)
        fr_tokens = re.findall(r"\S+|\s+", full_translated_text)

        cues = self._create_cues(source_words, fr_tokens)
        cues = self._adjust_timings(cues)

        return cues

    def _create_cues(self, source_words: List[Word], fr_tokens: List[str]) -> List[Cue]:
        cues: List[Cue] = []
        word_idx = 0
        token_idx = 0
        cue_idx = 1

        pbar = tqdm(total=len(source_words), desc="Création des sous-titres")

        while word_idx < len(source_words):
            # Define a time window based on source words
            start_word = source_words[word_idx]
            start_t = start_word.start

            # Greedily expand window until a significant gap is found
            end_word_idx = word_idx
            while end_word_idx + 1 < len(source_words):
                next_word = source_words[end_word_idx + 1]
                if (next_word.start - source_words[end_word_idx].end) > 0.8:
                    break
                end_word_idx += 1

            end_word = source_words[end_word_idx]
            end_t = end_word.end

            # Adjust duration to be within min/max constraints
            duration = end_t - start_t
            if duration < self.min_duration:
                end_t = start_t + self.min_duration
            if duration > self.max_duration:
                end_t = start_t + self.max_duration

            # Fit as many French tokens as possible into this time window
            current_text = ""
            tokens_in_cue = 0
            while token_idx < len(fr_tokens):
                token = fr_tokens[token_idx]
                if not self._can_add_token(token, start_t, end_t, current_text):
                    break
                current_text += token
                tokens_in_cue += 1
                token_idx += 1

            current_text = normalize_whitespace(current_text)
            if current_text:
                lines = split_balanced_two_lines(current_text, self.max_chars_per_line)
                formatted_text = "\n".join(lines[:self.max_lines])
                cues.append(Cue(cue_idx, start_t, end_t, formatted_text))
                cue_idx += 1

            pbar.update(end_word_idx - word_idx + 1)
            word_idx = end_word_idx + 1

        pbar.close()

        # Handle any remaining text
        remainder = "".join(fr_tokens[token_idx:]).strip()
        if remainder:
            start_t = cues[-1].end + self.gap if cues else 0.0
            end_t = start_t + min(self.max_duration, max(self.min_duration, len(remainder) / self.max_cps))
            lines = split_balanced_two_lines(remainder, self.max_chars_per_line)
            formatted_text = "\n".join(lines[:self.max_lines])
            cues.append(Cue(cue_idx, start_t, end_t, formatted_text))

        return cues

    def _can_add_token(self, token: str, start_t: float, end_t: float, current_text: str) -> bool:
        new_text = normalize_whitespace(current_text + token)
        duration = max(self.min_duration, end_t - start_t)

        if len(new_text) > self.max_chars_per_line * self.max_lines:
            return False

        cps = len(new_text) / duration
        if cps > self.max_cps:
            return False

        return True

    def _adjust_timings(self, cues: List[Cue]) -> List[Cue]:
        adjusted_cues = []
        for i, cue in enumerate(cues):
            start = cue.start
            end = cue.end

            # Ensure no overlap and respect minimum gap
            if i > 0:
                prev_end = adjusted_cues[i-1].end
                if start < prev_end + self.gap:
                    shift = (prev_end + self.gap) - start
                    start += shift
                    end += shift

            # Enforce min/max duration
            duration = end - start
            if duration < self.min_duration:
                end = start + self.min_duration
            if duration > self.max_duration:
                end = start + self.max_duration

            adjusted_cues.append(Cue(cue.idx, start, end, cue.text))
        return adjusted_cues

    def write_srt(self, cues: List[Cue], out_path: str):
        subs = [
            srt.Subtitle(
                index=c.idx,
                start=srt.srt_timestamp_to_timedelta(hhmmssmmm(c.start)),
                end=srt.srt_timestamp_to_timedelta(hhmmssmmm(c.end)),
                content=c.text,
            )
            for c in cues
        ]

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(srt.compose(subs))
            logger.info(f"Fichier SRT sauvegardé : {out_path}")
        except Exception as e:
            logger.error(f"Impossible d'écrire le fichier SRT : {e}")
