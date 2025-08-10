import os
import json
import typer
import logging
from typing import Optional, List, cast
from pathlib import Path
from dataclasses import asdict
import subprocess
import torch

from . import utils
from .downloader import YouTubeAudioDownloader
from .transcriber import Transcriber
from .translator import LLMTranslator
from .srt_builder import SRTBuilder

app = typer.Typer(
    help="Outil de transcription et de traduction de vidéos YouTube en SRT, 100% local.",
    add_completion=False,
    pretty_exceptions_show_locals=False
)

logger = logging.getLogger("emanet")

def _main_process(
    workdir: Path,
    outdir: Path,
    transcriber: Transcriber,
    translator: LLMTranslator,
    builder: SRTBuilder,
    entry: dict,
    glossary: dict,
    save_debug_json: bool,
    language: str,
):
    """Cœur du traitement pour une seule entrée vidéo."""
    title = entry["title"]
    vid_id = entry["id"]
    idx = entry["index"]
    media_path = Path(entry["m4a"])

    logger.info(f"--- Début du traitement de la vidéo #{idx}: {title} ---")

    base_slug = f"{idx:03d}-{utils.safe_slug(title)}-{vid_id}"
    wav_path = workdir / "wav" / f"{base_slug}.wav"

    if not media_path.exists():
        logger.warning(f"Fichier audio non trouvé pour '{title}', on passe : {media_path}")
        return

    # 1. Convert to WAV
    downloader = YouTubeAudioDownloader(str(workdir))
    downloader.convert_to_wav(str(media_path), str(wav_path))

    # 2. Transcription
    segments = transcriber.transcribe(str(wav_path), language=language)
    if not segments:
        logger.warning(f"Aucun segment transcrit pour '{title}', on passe.")
        return

    if save_debug_json:
        dbg_json_path = outdir / f"{base_slug}.segments.{language}.json"
        utils.ensure_dir(str(outdir))
        with open(dbg_json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in segments], f, ensure_ascii=False, indent=2)
        logger.debug(f"Segments de débogage sauvegardés dans {dbg_json_path}")

    # 3. Translation
    translated_blocks = translator.translate_segments(segments)
    if not translated_blocks:
        logger.warning(f"La traduction a échoué pour '{title}', on passe.")
        return

    processed_blocks = [utils.apply_glossary(utils.french_typography_fix(b), glossary) for b in translated_blocks if b.strip()]

    # 4. SRT Building
    cues = builder.build_from_segments(segments, processed_blocks)
    if not cues:
        logger.warning(f"Impossible de générer les sous-titres pour '{title}', on passe.")
        return

    # 5. Write SRT file
    srt_path = outdir / f"{base_slug}.fr.srt"
    builder.write_srt(cues, str(srt_path))

    logger.info(f"--- Traitement de la vidéo '{title}' terminé. ---")


@app.command(name="process", help="Lance le processus complet sur une playlist ou des fichiers locaux.")
def process_command(
    playlist_url: Optional[str] = typer.Option(None, "--playlist-url", "-p", help="URL de la playlist YouTube à traiter."),
    local_files: Optional[List[Path]] = typer.Option(None, "--local-files", "-l", help="Chemin vers un ou plusieurs fichiers audio/vidéo locaux.", exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    output_dir: Path = typer.Option("output", "--output-dir", "-o", help="Dossier de sortie pour les fichiers SRT.", file_okay=False, resolve_path=True),
    work_dir: Path = typer.Option("workdir", "--work-dir", "-w", help="Dossier de travail pour les fichiers intermédiaires (cache).", file_okay=False, resolve_path=True),
    cookies: Optional[Path] = typer.Option(None, "--cookies", help="Chemin vers le fichier cookies.txt pour yt-dlp.", exists=True, dir_okay=False, resolve_path=True),
    glossary: Optional[Path] = typer.Option(None, "--glossary", help="Chemin vers le glossaire (YAML ou key=value).", exists=True, dir_okay=False, resolve_path=True),

    language: str = typer.Option("fr", "--language", "-lang", help="Code de la langue de l'audio (ex: fr, en, es)."),
    asr_model: str = typer.Option("mistralai/Voxtral-Mini-3B-2507", "--asr-model", help="Modèle ASR (Voxtral) à utiliser."),
    asr_quant: Optional[str] = typer.Option(None, "--asr-quant", help="Quantification du modèle ASR (int4, int8) - GPU seulement."),

    llm_model: str = typer.Option("mistralai/Magistral-Small-2507", "--llm-model", help="Modèle LLM pour la traduction."),
    llm_quant: str = typer.Option("int4", "--llm-quant", help="Quantification du LLM (int4, int8, fp16)."),
    llm_window_sec: int = typer.Option(120, "--llm-window-sec", help="Taille de la fenêtre de contexte en secondes pour le LLM."),
    llm_batch_size: int = typer.Option(8, "--llm-batch-size", help="Taille du lot pour la traduction par le LLM."),
    style: str = typer.Option("neutral", "--style", help="Style de traduction (neutral, formal, informal)."),

    max_cps: float = typer.Option(17.0, "--max-cps", help="Caractères par seconde maximum pour les sous-titres."),
    max_chars_per_line: int = typer.Option(42, "--max-chars", help="Caractères maximum par ligne de sous-titre."),

    save_debug_json: bool = typer.Option(False, "--save-debug-json", help="Sauvegarder les segments transcrits en JSON pour le débogage."),
    debug: bool = typer.Option(False, "--debug", help="Activer les logs de débogage détaillés.")
):
    """
    Commande principale pour transcrire et traduire une playlist YouTube ou des fichiers locaux.
    """
    utils.setup_logging(debug)

    if not playlist_url and not local_files:
        logger.error("Erreur : Vous devez fournir une URL de playlist (--playlist-url) ou des fichiers locaux (--local-files).")
        raise typer.Exit(code=1)

    utils.ensure_dir(str(output_dir))
    utils.ensure_dir(str(work_dir))
    utils.ensure_dir(str(work_dir / "wav"))
    utils.ensure_dir(str(work_dir / "yt"))

    glossary_data = utils.load_glossary(str(glossary)) if glossary else {}

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transcriber = Transcriber(
            model_name=asr_model,
            device=device,
            quant=cast(Optional[str], asr_quant)
        )
        translator = LLMTranslator(
            model_name=llm_model, quant=llm_quant, window_sec=llm_window_sec,
            style=style, batch_size=llm_batch_size
        )
        builder = SRTBuilder(max_chars_per_line=max_chars_per_line, max_cps=max_cps)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des modèles : {e}")
        if debug: logger.exception("Traceback de l'erreur d'initialisation:")
        raise typer.Exit(code=1)

    entries = []
    if playlist_url:
        ytdl = YouTubeAudioDownloader(str(work_dir / "yt"), cookies=str(cookies) if cookies else None)
        entries.extend(ytdl.download_playlist(playlist_url))

    if local_files:
        for i, f in enumerate(local_files, start=len(entries) + 1):
            entries.append({"id": utils.safe_slug(f.stem), "index": i, "title": f.stem, "m4a": f})

    for entry in entries:
        try:
            _main_process(
                workdir=work_dir, outdir=output_dir, transcriber=transcriber,
                translator=translator, builder=builder, entry=entry,
                glossary=glossary_data, save_debug_json=save_debug_json,
                language=language
            )
        except Exception as e:
            logger.error(f"Une erreur critique est survenue lors du traitement de '{entry['title']}': {e}")
            if debug: logger.exception("Traceback de l'erreur:")
            continue

    logger.info("Toutes les tâches sont terminées.")

@app.command(name="health-check", help="Vérifie si l'environnement est correctement configuré.")
def health_check_command(debug: bool = typer.Option(False, "--debug", help="Activer les logs de débogage détaillés.")):
    """Exécute une série de tests pour valider l'installation et la configuration."""
    utils.setup_logging(debug)
    logger.info("Lancement du test d'intégrité...")
    has_failed = False

    def _check(name, func):
        nonlocal has_failed
        try:
            logger.info(f"--- Test: {name} ---")
            func()
            logger.info(f"✅ {name}: Succès")
            return True
        except Exception as e:
            logger.error(f"❌ {name}: Échec. Erreur: {e}")
            if debug: logger.exception("Traceback du test:")
            has_failed = True
            return False

    _check("Dépendances Python", lambda: __import__('torch') and __import__('transformers') and __import__('yt_dlp'))
    _check("Installation FFmpeg", lambda: utils.run_cmd(["ffmpeg", "-version"], check=True))

    # TODO: Réactiver ce test avec un petit modèle compatible Voxtral si disponible.
    # Le test ASR est commenté pour éviter de télécharger un gros modèle (Voxtral-Mini)
    # juste pour un test d'intégrité.
    # def asr_test():
    #     transcriber = Transcriber(model_name="mistralai/Voxtral-Mini-3B-2507")
    #     dummy_wav = Path("dummy_silent.wav")
    #     if not dummy_wav.exists():
    #         utils.run_cmd(["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", "-q:a", "9", "-acodec", "pcm_s16le", str(dummy_wav)], check=True)
    #     _ = transcriber.transcribe(str(dummy_wav), language="en")
    #     dummy_wav.unlink()
    # _check("Chargement et inférence ASR", asr_test)

    def llm_test():
        # Utilise distilgpt2 qui est un petit modèle de génération plus robuste pour ce test
        translator = LLMTranslator(model_name="distilgpt2", quant="fp16", batch_size=1)
        prompts = translator._build_prompts(["Ceci est un test."])
        results = translator._generate_batch(prompts)
        if not results or not results[0]:
            raise RuntimeError("La génération de texte par le LLM a échoué ou a retourné une chaîne vide.")
    _check("Chargement et inférence LLM", llm_test)

    def yt_test():
        with __import__('yt_dlp').YoutubeDL({"quiet": True, "skip_download": True, "ignoreerrors": True}) as ydl:
            res = ydl.extract_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ", download=False)
            if not res: raise RuntimeError("La récupération des métadonnées a échoué.")
    _check("Connectivité YouTube", yt_test)

    logger.info("-" * 20)
    if has_failed:
        logger.error("❌ Le test d'intégrité a échoué. Veuillez corriger les erreurs ci-dessus.")
        raise typer.Exit(code=1)
    else:
        logger.info("✅ Le test d'intégrité est réussi. L'environnement semble correctement configuré.")

if __name__ == "__main__":
    app()
