import os
import re
import yaml
import logging
import unicodedata
import subprocess
from typing import List, Optional, Dict

logger = logging.getLogger("emanet")

def run_cmd(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    logger.debug("CMD: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def safe_slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text or "video"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def hhmmssmmm(seconds: float) -> str:
    td_ms = int(seconds * 1000)
    hours = td_ms // 3600000
    minutes = (td_ms % 3600000) // 60000
    secs = (td_ms % 60000) // 1000
    ms = td_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def french_typography_fix(text: str) -> str:
    # Ajoute un espace avant ; : ! ?
    text = re.sub(r"\s*([;:!?])", r" \1", text)
    return text.strip()

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def split_balanced_two_lines(text: str, max_chars_per_line: int = 42) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars_per_line:
        return [text]
    mid = len(text) // 2
    left = text.rfind(" ", 0, mid)
    right = text.find(" ", mid)
    cut = -1
    candidates = [c for c in [left, right] if c != -1]
    if candidates:
        cut = min(candidates, key=lambda c: abs(len(text[:c]) - len(text[c + 1 :])))
    if cut == -1:
        cut = max_chars_per_line
    l1 = text[:cut].strip()
    l2 = text[cut:].strip()
    if len(l1) > max_chars_per_line:
        l1 = l1[:max_chars_per_line].rstrip()
    if len(l2) > max_chars_per_line:
        l2 = l2[:max_chars_per_line].rstrip()
    return [l1, l2]

def load_glossary(glossary_path: Optional[str]) -> Dict[str, str]:
    if not glossary_path or not os.path.isfile(glossary_path):
        return {}
    with open(glossary_path, "r", encoding="utf-8") as f:
        if glossary_path.endswith(".yaml") or glossary_path.endswith(".yml"):
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

def apply_glossary(text: str, glossary: Dict[str, str]) -> str:
    for src, tgt in glossary.items():
        if not src or not tgt:
            continue
        pattern = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
        text = pattern.sub(lambda m: match_case(m.group(0), tgt), text)
    return text

def match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.title()
    if source.islower():
        return target.lower()
    return target

def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
