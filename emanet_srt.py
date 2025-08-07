#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import srt
import json
import time
import yaml
import logging
import unicodedata
import argparse
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict

from tqdm import tqdm

# --- YouTube download ---
try:
    from yt_dlp import YoutubeDL
except Exception:
    YoutubeDL = None

# --- ASR ---
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

# --- Optional alignment ---
try:
    import whisperx
    HAS_WHISPERX = True
except Exception:
    HAS_WHISPERX = False

# --- Transformers / Torch ---
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,  # gardé pour fallback NLLB si besoin
)

# --------------------------
# Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("emanet")

# --------------------------
# Data structures
# --------------------------
@dataclass
class Word:
    text: str
    start: float
    end: float

@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None

@dataclass
class Cue:
    idx: int
    start: float
    end: float
    text: str

# --------------------------
# Utils
# --------------------------
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

# --------------------------
# YouTube Downloader
# --------------------------
class YouTubeAudioDownloader:
    def __init__(self, workdir: str, cookies: Optional[str] = None):
        self.workdir = workdir
        self.cookies = cookies
        ensure_dir(workdir)
        if YoutubeDL is None:
            raise RuntimeError("yt-dlp n'est pas installé.")

    def download_playlist(self, url: str) -> List[Dict]:
        outtmpl = os.path.join(self.workdir, "%(playlist_index)s-%(id)s.%(ext)s")
        ydl_opts = {
            "ignoreerrors": True,
            "outtmpl": outtmpl,
            "format": "bestaudio/best",
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"},
            ],
            "noprogress": True,
            "quiet": True,
        }
        if self.cookies and os.path.isfile(self.cookies):
            ydl_opts["cookiefile"] = self.cookies

        logger.info("Extraction de la playlist…")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        entries = info.get("entries", []) if info else []
        results = []
        for e in entries:
            if not e:
                continue
            vid = e.get("id")
            idx = e.get("playlist_index")
            title = e.get("title") or f"video-{vid}"
            m4a_path = os.path.join(self.workdir, f"{idx}-{vid}.m4a")
            if not os.path.exists(m4a_path):
                for fname in os.listdir(self.workdir):
                    if fname.startswith(f"{idx}-{vid}."):
                        m4a_path = os.path.join(self.workdir, fname)
                        break
            results.append({"id": vid, "index": idx, "title": title, "m4a": m4a_path})
        logger.info("Playlist: %d entrées", len(results))
        return results

    def convert_to_wav(self, in_path: str, out_wav: str, sr: int = 16000):
        ensure_dir(os.path.dirname(out_wav))
        if os.path.exists(out_wav):
            return
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", str(sr), "-vn", out_wav]
        run_cmd(cmd)

# --------------------------
# ASR (faster-whisper + option WhisperX)
# --------------------------
class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        model_dir: Optional[str] = None,
        use_whisperx_align: bool = True,
        beam_size: int = 5,
        best_of: int = 5,
    ):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper n'est pas installé.")
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type, download_root=model_dir
        )
        self.device = device
        self.model_dir = model_dir
        self.use_whisperx_align = use_whisperx_align and HAS_WHISPERX
        self.beam_size = beam_size
        self.best_of = best_of

    def transcribe(self, audio_path: str, language: str = "tr") -> List[Segment]:
        logger.info("Transcription (faster-whisper) de %s", os.path.basename(audio_path))
        segments = []
        t0 = time.time()
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
        for seg in it:
            words = None
            if seg.words:
                words = [Word(w.word, w.start, w.end) for w in seg.words if w.word.strip()]
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
        logger.info("Transcription terminée en %.1fs — %d segments", time.time() - t0, len(segments))

        # Filtrage léger
        filtered = []
        for s in segments:
            if s.no_speech_prob is not None and s.no_speech_prob > 0.6:
                continue
            if s.avg_logprob is not None and s.avg_logprob < -1.25:
                continue
            filtered.append(s)
        segments = filtered

        if self.use_whisperx_align:
            try:
                segments = self._align_with_whisperx(audio_path, segments, language)
            except Exception as e:
                logger.warning("Alignement WhisperX échoué (%s) — on continue sans alignement.", e)

        return segments

    def _align_with_whisperx(self, audio_path: str, segments: List[Segment], language: str) -> List[Segment]:
        logger.info("Alignement mot-à-mot (WhisperX)…")
        device = self.device if torch.cuda.is_available() else "cpu"
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
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
            for w in wx.get("words", []):
                if "start" in w and "end" in w and "word" in w:
                    if w["start"] is None or w["end"] is None:
                        continue
                    words.append(Word(w["word"], float(w["start"]), float(w["end"])))
            aligned.append(Segment(start=wx["start"], end=wx["end"], text=wx["text"].strip(), words=words))
        logger.info("Alignement WhisperX OK — %d segments", len(aligned))
        return aligned

# --------------------------
# SRT builder (re-segmentation propre)
# --------------------------
class SRTBuilder:
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
        # 1) Fenêtres temporelles issues des mots source
        words = []
        for s in segments:
            if s.words:
                for w in s.words:
                    words.append(Word(text=w.text, start=w.start, end=w.end))
            else:
                words.append(Word(text=s.text, start=s.start, end=s.end))
        words.sort(key=lambda w: (w.start, w.end))

        # 2) Texte FR global
        raw_text = " ".join([normalize_whitespace(t) for t in translated_texts])
        raw_text = french_typography_fix(raw_text)
        fr_tokens = re.findall(r"\S+|\s+", raw_text)
        token_idx = 0

        cues: List[Cue] = []
        cue_idx = 1

        if not words:
            return [Cue(1, 0.0, max(self.min_duration, 2.0), raw_text)]

        def can_add(token: str, start_t: float, end_t: float, current_text: str) -> bool:
            new_text = normalize_whitespace(current_text + token)
            dur = max(self.min_duration, end_t - start_t)
            if len(new_text) > self.max_chars_per_line * self.max_lines:
                return False
            cps = len(new_text) / max(dur, 0.001)
            if cps > self.max_cps:
                return False
            if dur > self.max_duration:
                return False
            return True

        i = 0
        while i < len(words) and token_idx < len(fr_tokens):
            win_words = [words[i]]
            j = i + 1
            while j < len(words):
                if words[j].start - win_words[-1].end > 0.8:
                    break
                win_words.append(words[j])
                j += 1

            start_t = win_words[0].start
            end_t = win_words[-1].end
            dur = end_t - start_t
            if dur < self.min_duration:
                end_t = start_t + self.min_duration
            if (end_t - start_t) > self.max_duration:
                end_t = start_t + self.max_duration

            cur = ""
            while token_idx < len(fr_tokens):
                tok = fr_tokens[token_idx]
                if not can_add(tok, start_t, end_t, cur):
                    break
                cur += tok
                token_idx += 1

            cur = normalize_whitespace(cur)
            if not cur and token_idx < len(fr_tokens):
                end_t = start_t + max(self.min_duration, (end_t - start_t) + 0.5)
                while token_idx < len(fr_tokens):
                    tok = fr_tokens[token_idx]
                    if not can_add(tok, start_t, end_t, cur):
                        break
                    cur += tok
                    token_idx += 1
                cur = normalize_whitespace(cur)

            if cur:
                lines = split_balanced_two_lines(cur, self.max_chars_per_line)
                if len(lines) > 2:
                    lines = lines[:2]
                cur = "\n".join(lines)
                if cues:
                    start_t = max(start_t, cues[-1].end + self.gap)
                    if end_t <= start_t:
                        end_t = start_t + self.min_duration
                cues.append(Cue(cue_idx, start_t, end_t, cur))
                cue_idx += 1

            i = j

        remainder = "".join(fr_tokens[token_idx:]).strip()
        if remainder:
            start_t = cues[-1].end + self.gap if cues else 0.0
            end_t = start_t + min(self.max_duration, max(self.min_duration, len(remainder) / self.max_cps))
            lines = split_balanced_two_lines(remainder, self.max_chars_per_line)
            cur = "\n".join(lines[:2])
            cues.append(Cue(cue_idx, start_t, end_t, cur))

        # Nettoyage timings
        for k in range(1, len(cues)):
            if cues[k].start < cues[k - 1].end + self.gap:
                shift = (cues[k - 1].end + self.gap) - cues[k].start
                cues[k] = Cue(cues[k].idx, cues[k].start + shift, cues[k].end + shift, cues[k].text)
            dur = cues[k].end - cues[k].start
            if dur < self.min_duration:
                cues[k] = Cue(cues[k].idx, cues[k].start, cues[k].start + self.min_duration, cues[k].text)
            if dur > self.max_duration:
                cues[k] = Cue(cues[k].idx, cues[k].start, cues[k].start + self.max_duration, cues[k].text)

        return cues

    def write_srt(self, cues: List[Cue], out_path: str):
        subs = []
        for c in cues:
            subs.append(
                srt.Subtitle(
                    index=c.idx,
                    start=srt.srt_timestamp_to_timedelta(hhmmssmmm(c.start)),
                    end=srt.srt_timestamp_to_timedelta(hhmmssmmm(c.end)),
                    content=c.text,
                )
            )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(subs))
        logger.info("Écrit: %s", out_path)

# --------------------------
# Translators
# --------------------------
class LLMTranslator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        quant: str = "int4",  # int4 | int8 | fp16
        window_sec: int = 120,
        style: str = "neutral",  # neutral | formal | informal
        max_new_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.quant = quant
        self.window_sec = window_sec
        self.style = style
        self.max_new_tokens = max_new_tokens

        logger.info("Chargement LLM (%s, %s)…", model_name, quant)
        load_kwargs = {"device_map": "auto"}
        dtype = torch.bfloat16 if torch.cuda.is_available() else None

        if quant == "int4":
            try:
                load_kwargs.update(dict(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
            except Exception:
                logger.warning("Quantification 4-bit indisponible, fallback FP16/ bfloat16.")
                load_kwargs["torch_dtype"] = dtype
        elif quant == "int8":
            try:
                load_kwargs.update(dict(load_in_8bit=True))
            except Exception:
                logger.warning("Quantification 8-bit indisponible, fallback FP16/ bfloat16.")
                load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **load_kwargs)
        self.has_chat_template = hasattr(self.tokenizer, "apply_chat_template")

    def _system_prompt(self) -> str:
        style_map = {
            "neutral": "registre naturel, neutre",
            "formal": "registre soutenu, vouvoiement",
            "informal": "registre familier, tutoiement",
        }
        style_txt = style_map.get(self.style, "registre naturel, neutre")
        return (
            "Tu es un traducteur professionnel turc → français. "
            f"Règles: {style_txt}; rendu idiomatique et fluide; fidèle au sens; "
            "pas d'explications; n'ajoute rien; conserve les sauts de ligne du texte source."
        )

    def _build_prompt(self, tr_text: str) -> str:
        instr = (
            "Traduis le texte turc ci-dessous en français.\n"
            "Exigences:\n"
            "- Sens fidèle, naturel et idiomatique\n"
            "- Pas de notes ni d'explications\n"
            "- Conserve les retours à la ligne\n\n"
            "Texte (turc):\n"
            f"{tr_text}\n\n"
            "Réponse (français) uniquement:"
        )
        if self.has_chat_template:
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": instr},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return f"[SYSTEM]\n{self._system_prompt()}\n\n[USER]\n{instr}\n\n[ASSISTANT]\n"

    def _generate(self, prompt: str) -> str:
        tok = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **tok,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # déterministe
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):]
        text = re.sub(r"^(Traduction|French|FR|Réponse)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        return text

    def translate_with_context(self, segments: List[Segment]) -> List[str]:
        # Grouper par fenêtres temporelles ~window_sec
        windows: List[List[Segment]] = []
        cur: List[Segment] = []
        cur_start = segments[0].start if segments else 0.0
        for s in segments:
            if not cur:
                cur = [s]
                cur_start = s.start
                continue
            if (s.end - cur_start) <= self.window_sec:
                cur.append(s)
            else:
                windows.append(cur)
                cur = [s]
                cur_start = s.start
        if cur:
            windows.append(cur)

        logger.info("LLM: %d fenêtres contexte (~%ds chacune)", len(windows), self.window_sec)

        fr_blocks: List[str] = []
        for win in tqdm(windows, desc="LLM TR→FR"):
            tr_text = "\n".join([normalize_whitespace(s.text) for s in win if s.text.strip()])
            if not tr_text.strip():
                continue
            fr = self._generate(self._build_prompt(tr_text))
            fr_blocks.append(fr)

        return fr_blocks

# --------------------------
# Pipeline
# --------------------------
def process_video(
    entry: Dict,
    workdir: str,
    outdir: str,
    transcriber: Transcriber,
    translator: LLMTranslator,
    glossary: Dict[str, str],
    builder: SRTBuilder,
    save_debug: bool = True,
):
    title = entry["title"]
    vid = entry["id"]
    idx = entry["index"]
    m4a = entry["m4a"]

    base_slug = f"{idx:03d}-{safe_slug(title)}-{vid}"
    wav_path = os.path.join(workdir, "wav", f"{base_slug}.wav")
    ensure_dir(os.path.dirname(wav_path))

    if not os.path.exists(m4a):
        logger.warning("Fichier audio non trouvé pour %s, on passe.", title)
        return
    YouTubeAudioDownloader(workdir).convert_to_wav(m4a, wav_path, sr=16000)

    segments = transcriber.transcribe(wav_path, language="tr")

    if save_debug:
        dbg_json = os.path.join(outdir, f"{base_slug}.segments.tr.json")
        ensure_dir(outdir)
        with open(dbg_json, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "avg_logprob": s.avg_logprob,
                        "no_speech_prob": s.no_speech_prob,
                        "words": [{"text": w.text, "start": w.start, "end": w.end} for w in (s.words or [])],
                    }
                    for s in segments
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

    # Traduction locale (LLM context-aware)
    logger.info("Traduction TR → FR (LLM)…")
    fr_blocks = translator.translate_with_context(segments)
    fr_blocks = [apply_glossary(french_typography_fix(b), glossary) for b in fr_blocks if b.strip()]

    cues = builder.build_from_segments(segments, fr_blocks)

    srt_path = os.path.join(outdir, f"{base_slug}.fr.srt")
    builder.write_srt(cues, srt_path)

    logger.info("Terminé: %s", srt_path)

def main():
    parser = argparse.ArgumentParser(description="Emanet TR→FR SRT (RunPod/B200, 100% local)")
    parser.add_argument("--playlist-url", type=str, required=False, help="URL de la playlist YouTube")
    parser.add_argument("--output-dir", type=str, default="output", help="Dossier de sortie SRT")
    parser.add_argument("--work-dir", type=str, default="workdir", help="Dossier de travail (cache)")
    parser.add_argument("--local-files", type=str, nargs="*", help="Fichiers audio/vidéo locaux")
    parser.add_argument("--cookies", type=str, default=None, help="cookies.txt pour yt-dlp (optionnel)")
    parser.add_argument("--glossary", type=str, default=None, help="Glossaire (yaml/yml ou key=value)")
    # ASR
    parser.add_argument("--asr-model", type=str, default="large-v3", help="Modèle faster-whisper (ex: large-v3)")
    parser.add_argument("--asr-compute-type", type=str, default="float16", help="float16 ou int8_float16")
    parser.add_argument("--no-align", action="store_true", help="Désactiver l'alignement WhisperX")
    # LLM (valeurs choisies par défaut, pas besoin de changer)
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm-quant", type=str, default="int4", choices=["int4", "int8", "fp16"])
    parser.add_argument("--llm-window-sec", type=int, default=120)
    parser.add_argument("--style", type=str, default="neutral", choices=["neutral", "formal", "informal"])
    # SRT
    parser.add_argument("--max-cps", type=float, default=17.0)
    parser.add_argument("--max-chars", type=int, default=42)
    parser.add_argument("--save-debug", action="store_true", help="Sauver JSON de debug")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.work_dir)

    glossary = load_glossary(args.glossary)

    transcriber = Transcriber(
        model_size=args.asr_model,
        compute_type=args.asr_compute_type,
        use_whisperx_align=not args.no_align,
        beam_size=5,
        best_of=5,
    )

    translator = LLMTranslator(
        model_name=args.llm_model,
        quant=args.llm_quant,
        window_sec=args.llm_window_sec,
        style=args.style,
        max_new_tokens=2048,
    )

    builder = SRTBuilder(
        max_chars_per_line=args.max_chars,
        max_lines=2,
        max_cps=args.max_cps,
        min_duration=1.0,
        max_duration=7.0,
        gap=0.08,
    )

    if args.playlist_url:
        if YoutubeDL is None:
            raise RuntimeError("yt-dlp non installé.")
        ytdl = YouTubeAudioDownloader(os.path.join(args.work_dir, "yt"), cookies=args.cookies)
        entries = ytdl.download_playlist(args.playlist_url)
        for e in entries:
            process_video(
                e,
                workdir=args.work_dir,
                outdir=args.output_dir,
                transcriber=transcriber,
                translator=translator,
                glossary=glossary,
                builder=builder,
                save_debug=args.save_debug,
            )

    if args.local_files:
        for idx, path in enumerate(args.local_files, start=1):
            if not os.path.exists(path):
                logger.warning("Fichier introuvable: %s", path)
                continue
            base = os.path.splitext(os.path.basename(path))[0]
            fake_entry = {"id": base, "index": idx, "title": base, "m4a": path}
            process_video(
                fake_entry,
                workdir=args.work_dir,
                outdir=args.output_dir,
                transcriber=transcriber,
                translator=translator,
                glossary=glossary,
                builder=builder,
                save_debug=args.save_debug,
            )

    if not args.playlist_url and not args.local_files:
        logger.warning("Aucune source fournie. Utilisez --playlist-url ou --local-files.")

if __name__ == "__main__":
    main()
