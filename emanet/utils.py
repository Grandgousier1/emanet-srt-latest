import os
import re
import yaml
import logging
import unicodedata
import subprocess
from typing import List, Optional, Dict

logger = logging.getLogger("emanet")


def run_cmd(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    """Exécute une commande shell et retourne le résultat."""
    logger.debug("CMD: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def safe_slug(text: str) -> str:
    """Convertit un texte en un 'slug' sûr pour les noms de fichiers."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text or "video"


def ensure_dir(path: str):
    """S'assure qu'un dossier existe, le crée sinon."""
    os.makedirs(path, exist_ok=True)


def to_srt_timestamp(seconds: float) -> str:
    """Convertit des secondes en format d'horodatage SRT (HH:MM:SS,mmm)."""
    td_ms = int(seconds * 1000)
    hours = td_ms // 3600000
    minutes = (td_ms % 3600000) // 60000
    secs = (td_ms % 60000) // 1000
    ms = td_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def french_typography_fix(text: str) -> str:
    """Applique des corrections typographiques françaises de base."""
    # Ajoute des espaces avant les ponctuations doubles et les guillemets fermants
    text = re.sub(r'\s*([;:!?»])', r' \1', text)
    # Ajoute des espaces après les guillemets ouvrants
    text = re.sub(r'([«])\s*', r'\1 ', text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Remplace les séquences d'espaces par un seul espace et nettoie les bords."""
    return re.sub(r"\s+", " ", text).strip()


def split_balanced_two_lines(text: str, max_chars_per_line: int = 42) -> List[str]:
    """Divise un texte en deux lignes de longueur à peu près équilibrée."""
    text = text.strip()
    if len(text) <= max_chars_per_line:
        return [text]

    mid = len(text) // 2
    # Cherche l'espace le plus proche du milieu
    left = text.rfind(" ", 0, mid)
    right = text.find(" ", mid)

    cut = -1
    candidates = [c for c in [left, right] if c != -1]
    if candidates:
        # Choisit le candidat qui équilibre le mieux les longueurs des deux lignes
        cut = min(candidates, key=lambda c: abs(len(text[:c]) - len(text[c+1:])))

    # Si aucun espace n'est trouvé, on coupe brutalement à la longueur max
    if cut == -1:
        cut = max_chars_per_line

    l1 = text[:cut].strip()
    l2 = text[cut:].strip()

    # S'assure qu'aucune ligne ne dépasse la limite
    if len(l1) > max_chars_per_line:
        l1 = l1[:max_chars_per_line].rstrip()
    if len(l2) > max_chars_per_line:
        l2 = l2[:max_chars_per_line].rstrip()

    return [l1, l2]


def load_glossary(glossary_path: Optional[str]) -> Dict[str, str]:
    """Charge un glossaire depuis un fichier YAML ou texte (format: key=value)."""
    if not glossary_path or not os.path.isfile(glossary_path):
        return {}

    try:
        with open(glossary_path, "r", encoding="utf-8") as f:
            if glossary_path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f) or {}
            else:
                data = {}
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip()

        return {k.strip(): v.strip() for k, v in data.items() if k and v}
    except Exception as e:
        logger.error(f"Impossible de charger le glossaire depuis {glossary_path}: {e}")
        return {}


def apply_glossary(text: str, glossary: Dict[str, str]) -> str:
    """Applique un glossaire à un texte, en respectant la casse du mot source."""
    if not glossary or not text:
        return text

    # Crée une version du glossaire avec des clés en minuscules pour une recherche insensible à la casse
    glossary_lower = {k.lower(): v for k, v in glossary.items()}

    # Construit une seule expression régulière pour tous les mots du glossaire
    # \b assure qu'on ne remplace que des mots entiers
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in glossary.keys()) + r')\b',
        flags=re.IGNORECASE
    )

    def replace_func(match):
        source_word = match.group(0)
        replacement = glossary_lower.get(source_word.lower())
        return match_case(source_word, replacement) if replacement else source_word

    return pattern.sub(replace_func, text)


def match_case(source: str, target: str) -> str:
    """Applique la casse de la chaîne source à la chaîne cible."""
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.title()
    # Pour les mots en milieu de phrase, on met en minuscule.
    if source and source[0].islower():
        return target.lower()
    return target


def setup_logging(debug: bool = False):
    """Configure le logging pour l'application."""
    level = logging.DEBUG if debug else logging.INFO

    # Evite d'ajouter plusieurs handlers si la fonction est appelée plusieurs fois
    if logging.root.handlers:
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

class SileroVAD:
    """Classe pour gérer le modèle Silero VAD et éviter les re-téléchargements."""
    _model = None
    _utils = None

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            logger.info("Chargement du modèle Silero VAD...")
            try:
                import torch
                # Le modèle sera mis en cache par torch.hub dans ~/.cache/torch/hub
                cls._model, cls._utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False  # Utilise PyTorch pour compatibilité CPU
                )
                logger.info("Modèle Silero VAD chargé.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle Silero VAD: {e}")
                raise

    @classmethod
    def get_speech_timestamps(cls, audio_path: str, min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 100):
        """
        Détecte les segments de parole dans un fichier audio.

        Args:
            audio_path (str): Chemin vers le fichier audio (WAV, 16kHz mono).
            min_speech_duration_ms (int): Durée minimale d'un segment de parole.
            min_silence_duration_ms (int): Durée minimale d'un silence entre les segments.

        Returns:
            List[Dict[str, float]]: Liste de dictionnaires avec les clés 'start' et 'end' en secondes.
        """
        import torch
        import torchaudio

        cls._load_model()
        if cls._model is None:
            return []

        try:
            wav, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                # Resample si nécessaire
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                wav = resampler(wav)

            if wav.shape[0] > 1: # Si stéréo, on prend la moyenne
                wav = torch.mean(wav, dim=0, keepdim=True)

        except Exception as e:
            logger.error(f"Impossible de charger ou de traiter le fichier audio {audio_path}: {e}")
            return []

        (get_speech_timestamps_func, _, _, _, _) = cls._utils

        timestamps = get_speech_timestamps_func(
            wav,
            cls._model,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms
        )

        # Conversion des échantillons en secondes
        for ts in timestamps:
            ts['start'] /= 16000
            ts['end'] /= 16000

        logger.info(f"VAD a trouvé {len(timestamps)} segments de parole dans {os.path.basename(audio_path)}.")
        return timestamps
