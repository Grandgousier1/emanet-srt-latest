import re
import srt
import logging
from typing import List

from .datastructures import Segment, Cue, Word
from .utils import to_srt_timestamp, normalize_whitespace, french_typography_fix, split_balanced_two_lines

logger = logging.getLogger("emanet")

class SRTBuilder:
    """
    Construit un fichier SRT propre et bien formaté à partir de segments transcrits et de textes traduits.
    L'algorithme principal vise à synchroniser le texte traduit avec l'horodatage des mots d'origine.
    """
    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        max_cps: float = 17.0,
        min_duration: float = 1.0,
        max_duration: float = 7.0,
        gap_between_cues: float = 0.08,
    ):
        """
        Initialise le constructeur de SRT.

        Args:
            max_chars_per_line (int): Nombre maximum de caractères par ligne de sous-titre.
            max_lines (int): Nombre maximum de lignes par sous-titre.
            max_cps (float): Caractères par seconde maximum pour évaluer la lisibilité.
            min_duration (float): Durée minimale d'un sous-titre en secondes.
            max_duration (float): Durée maximale d'un sous-titre en secondes.
            gap_between_cues (float): Espace minimum en secondes entre deux sous-titres consécutifs.
        """
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.max_cps = max_cps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.gap = gap_between_cues

    def build_from_segments(self, segments: List[Segment], translated_texts: List[str]) -> List[Cue]:
        """
        Point d'entrée principal pour construire les sous-titres.

        Args:
            segments (List[Segment]): Segments transcrits de l'audio original.
            translated_texts (List[str]): Blocs de texte correspondants, traduits.

        Returns:
            List[Cue]: Une liste d'objets Cue prêts à être écrits dans un fichier SRT.
        """
        # 1. Obtenir tous les mots sources avec leur horodatage pour la référence temporelle.
        source_words = self._get_source_words(segments)
        if not source_words:
            logger.warning("Aucun mot source avec horodatage trouvé. Impossible de construire le SRT.")
            return []

        # 2. Concaténer et tokeniser le texte français traduit.
        full_translated_text = " ".join([normalize_whitespace(t) for t in translated_texts])
        full_translated_text = french_typography_fix(full_translated_text)
        fr_tokens = re.findall(r"\S+|\s+", full_translated_text)

        # 3. Créer les cues en se basant sur le timing des mots sources et le texte traduit.
        cues = self._create_cues(source_words, fr_tokens)

        # 4. Ajuster finement les timings pour éviter les chevauchements et respecter les durées.
        cues = self._adjust_timings(cues)

        return cues

    def _get_source_words(self, segments: List[Segment]) -> List[Word]:
        """Extrait une liste plate de mots horodatés à partir des segments."""
        source_words = []
        for s in segments:
            if s.words:
                source_words.extend(s.words)
            elif s.text.strip():  # Fallback sur le timing du segment si pas de mots
                source_words.append(Word(text=s.text, start=s.start, end=s.end))
        source_words.sort(key=lambda w: (w.start, w.end))
        return source_words

    def _create_cues(self, source_words: List[Word], fr_tokens: List[str]) -> List[Cue]:
        """Algorithme de création de sous-titres."""
        cues: List[Cue] = []
        word_idx, token_idx, cue_idx = 0, 0, 1

        pbar = tqdm(total=len(fr_tokens), desc="Création des sous-titres")

        while word_idx < len(source_words) and token_idx < len(fr_tokens):
            # 1. Définir une fenêtre temporelle à partir des mots sources.
            start_t = source_words[word_idx].start

            # 2. Étendre la fenêtre pour inclure les mots suivants jusqu'à une pause significative.
            # Le seuil de 0.8s est une heuristique pour détecter une pause dans le discours.
            end_word_idx = word_idx
            while end_word_idx + 1 < len(source_words):
                if (source_words[end_word_idx + 1].start - source_words[end_word_idx].end) > 0.8:
                    break
                end_word_idx += 1
            end_t = source_words[end_word_idx].end

            # 3. Ajuster la durée de la fenêtre pour respecter les contraintes min/max.
            duration = end_t - start_t
            if duration < self.min_duration: end_t = start_t + self.min_duration
            if duration > self.max_duration: end_t = start_t + self.max_duration

            # 4. Remplir la fenêtre avec autant de tokens traduits que possible.
            current_text = ""
            tokens_in_cue_count = 0
            temp_token_idx = token_idx
            while temp_token_idx < len(fr_tokens):
                token = fr_tokens[temp_token_idx]
                if not self._can_add_token(token, start_t, end_t, current_text):
                    break
                current_text += token
                tokens_in_cue_count += 1
                temp_token_idx += 1

            # 5. Créer le sous-titre s'il contient du texte.
            current_text = normalize_whitespace(current_text)
            if current_text:
                lines = split_balanced_two_lines(current_text, self.max_chars_per_line)
                formatted_text = "\n".join(lines[:self.max_lines])
                cues.append(Cue(cue_idx, start_t, end_t, formatted_text))
                cue_idx += 1
                token_idx += tokens_in_cue_count
                pbar.update(tokens_in_cue_count)

            word_idx = end_word_idx + 1

        pbar.close()

        # 6. Gérer le texte restant qui n'a pas pu être placé.
        remainder = "".join(fr_tokens[token_idx:]).strip()
        if remainder:
            start_t = cues[-1].end + self.gap if cues else 0.0
            end_t = start_t + min(self.max_duration, max(self.min_duration, len(remainder) / self.max_cps))
            lines = split_balanced_two_lines(remainder, self.max_chars_per_line)
            cues.append(Cue(cue_idx, start_t, end_t, "\n".join(lines[:self.max_lines])))

        return cues

    def _can_add_token(self, token: str, start_t: float, end_t: float, current_text: str) -> bool:
        """Vérifie si un token peut être ajouté à un sous-titre sans violer les contraintes."""
        new_text = normalize_whitespace(current_text + token)
        duration = max(self.min_duration, end_t - start_t)

        if len(new_text) > self.max_chars_per_line * self.max_lines: return False
        if (len(new_text) / duration) > self.max_cps: return False

        return True

    def _adjust_timings(self, cues: List[Cue]) -> List[Cue]:
        """Ajuste les timings des cues pour éviter les chevauchements et forcer les durées min/max."""
        for i, cue in enumerate(cues):
            # Assurer un écart minimal avec le sous-titre précédent
            if i > 0:
                prev_end = cues[i-1].end
                if cue.start < prev_end + self.gap:
                    shift = (prev_end + self.gap) - cue.start
                    cue.start += shift
                    cue.end += shift

            # Forcer les durées min/max
            duration = cue.end - cue.start
            if duration < self.min_duration:
                cue.end = cue.start + self.min_duration
            elif duration > self.max_duration:
                cue.end = cue.start + self.max_duration

        return cues

    def write_srt(self, cues: List[Cue], out_path: str):
        """
        Écrit une liste de Cues dans un fichier au format .srt.

        Args:
            cues (List[Cue]): La liste des sous-titres à écrire.
            out_path (str): Le chemin du fichier de sortie.
        """
        subs = [
            srt.Subtitle(
                index=c.idx,
                start=srt.srt_timestamp_to_timedelta(to_srt_timestamp(c.start)),
                end=srt.srt_timestamp_to_timedelta(to_srt_timestamp(c.end)),
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
