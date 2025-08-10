import os
import typer
import logging
from typing import Optional, List
from pathlib import Path

from . import utils
from .downloader import YouTubeAudioDownloader
from .transcriber import Transcriber
from .translator import LLMTranslator
from .srt_builder import SRTBuilder
from .datastructures import Segment, Cue

app = typer.Typer(
    help="Outil de transcription et de traduction de vidéos YouTube en SRT, 100% local.",
    add_completion=False
)

logger = logging.getLogger("emanet")

def main_process(
    workdir: Path,
    outdir: Path,
    transcriber: Transcriber,
    translator: LLMTranslator,
    builder: SRTBuilder,
    entry: dict,
    glossary: dict,
    save_debug_json: bool
):
    """Processes a single video entry (from playlist or local file)."""
    title = entry["title"]
    vid_id = entry["id"]
    idx = entry["index"]
    media_path = entry["m4a"] # Can be m4a from YouTube or any local media file

    logger.info(f"--- Début du traitement de la vidéo #{idx}: {title} ---")

    base_slug = f"{idx:03d}-{utils.safe_slug(title)}-{vid_id}"
    wav_path = workdir / "wav" / f"{base_slug}.wav"

    if not media_path.exists():
        logger.warning(f"Fichier audio non trouvé pour '{title}', on passe.")
        return

    # 1. Convert to WAV
    downloader = YouTubeAudioDownloader(str(workdir)) # only used for conversion here
    downloader.convert_to_wav(str(media_path), str(wav_path))

    # 2. Transcription
    segments = transcriber.transcribe(str(wav_path), language="tr")
    if not segments:
        logger.warning(f"Aucun segment transcrit pour '{title}', on passe.")
        return

    if save_debug_json:
        dbg_json_path = outdir / f"{base_slug}.segments.tr.json"
        utils.ensure_dir(str(outdir))
        with open(dbg_json_path, "w", encoding="utf-8") as f:
            import json
            json.dump([s.__dict__ for s in segments], f, ensure_ascii=False, indent=2, default=lambda o: o.__dict__)
        logger.debug(f"Segments de débogage sauvegardés dans {dbg_json_path}")


    # 3. Translation
    translated_blocks = translator.translate_segments(segments)
    if not translated_blocks:
        logger.warning(f"La traduction a échoué pour '{title}', on passe.")
        return

    # Apply glossary and typography fixes
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
    local_files: Optional[List[Path]] = typer.Option(None, "--local-files", "-l", help="Chemin vers un ou plusieurs fichiers audio/vidéo locaux.", exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option("output", "--output-dir", "-o", help="Dossier de sortie pour les fichiers SRT.", file_okay=False, resolve_path=True),
    work_dir: Path = typer.Option("workdir", "--work-dir", "-w", help="Dossier de travail pour les fichiers intermédiaires (cache).", file_okay=False, resolve_path=True),
    cookies: Optional[Path] = typer.Option(None, "--cookies", help="Chemin vers le fichier cookies.txt pour yt-dlp.", exists=True, dir_okay=False, resolve_path=True),
    glossary: Optional[Path] = typer.Option(None, "--glossary", help="Chemin vers le glossaire (YAML ou key=value).", exists=True, dir_okay=False, resolve_path=True),

    # ASR Options
    asr_model: str = typer.Option("large-v3", "--asr-model", help="Modèle faster-whisper à utiliser."),
    asr_compute_type: str = typer.Option("float16", "--asr-compute-type", help="Type de calcul pour le modèle ASR (ex: float16, int8_float16)."),
    no_align: bool = typer.Option(False, "--no-align", help="Désactiver l'alignement mot-à-mot avec WhisperX."),

    # LLM Options
    llm_model: str = typer.Option("mistralai/Magistral-Small-2507", "--llm-model", help="Modèle LLM à utiliser pour la traduction."),
    llm_quant: str = typer.Option("int4", "--llm-quant", help="Quantification du LLM (int4, int8, fp16)."),
    llm_window_sec: int = typer.Option(120, "--llm-window-sec", help="Taille de la fenêtre de contexte en secondes pour le LLM."),
    style: str = typer.Option("neutral", "--style", help="Style de traduction (neutral, formal, informal)."),

    # SRT Options
    max_cps: float = typer.Option(17.0, "--max-cps", help="Caractères par seconde maximum pour les sous-titres."),
    max_chars_per_line: int = typer.Option(42, "--max-chars", help="Caractères maximum par ligne de sous-titre."),

    save_debug_json: bool = typer.Option(False, "--save-debug-json", help="Sauvegarder les segments transcrits en JSON pour le débogage."),
    debug: bool = typer.Option(False, "--debug", help="Activer les logs de débogage détaillés.")
):
    utils.setup_logging(debug)

    if not playlist_url and not local_files:
        logger.error("Erreur : Vous devez fournir une URL de playlist (--playlist-url) ou des fichiers locaux (--local-files).")
        raise typer.Exit(code=1)

    # Create directories
    utils.ensure_dir(str(output_dir))
    utils.ensure_dir(str(work_dir))
    utils.ensure_dir(str(work_dir / "wav"))
    utils.ensure_dir(str(work_dir / "yt"))

    # Load resources
    glossary_data = utils.load_glossary(str(glossary)) if glossary else {}

    # Initialize components
    try:
        transcriber = Transcriber(
            model_name=asr_model,
            compute_type=asr_compute_type,
            use_whisperx_align=not no_align
        )
        translator = LLMTranslator(
            model_name=llm_model,
            quant=llm_quant,
            window_sec=llm_window_sec,
            style=style
        )
        builder = SRTBuilder(
            max_chars_per_line=max_chars_per_line,
            max_cps=max_cps
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des modèles : {e}")
        raise typer.Exit(code=1)

    entries = []
    if playlist_url:
        ytdl = YouTubeAudioDownloader(str(work_dir / "yt"), cookies=str(cookies) if cookies else None)
        yt_entries = ytdl.download_playlist(playlist_url)
        for e in yt_entries:
            e["m4a"] = Path(e["m4a"])
        entries.extend(yt_entries)

    if local_files:
        for i, f in enumerate(local_files, start=len(entries) + 1):
            entries.append({
                "id": utils.safe_slug(f.stem),
                "index": i,
                "title": f.stem,
                "m4a": f
            })

    # Process each entry
    for entry in entries:
        try:
            main_process(
                workdir=work_dir,
                outdir=output_dir,
                transcriber=transcriber,
                translator=translator,
                builder=builder,
                entry=entry,
                glossary=glossary_data,
                save_debug_json=save_debug_json
            )
        except Exception as e:
            logger.error(f"Une erreur critique est survenue lors du traitement de '{entry['title']}': {e}")
            if debug:
                logger.exception("Traceback de l'erreur:")
            continue # Continue with the next file

    logger.info("Toutes les tâches sont terminées.")

@app.command(name="health-check", help="Vérifie si l'environnement est correctement configuré.")
def health_check_command(
    debug: bool = typer.Option(False, "--debug", help="Activer les logs de débogage détaillés.")
):
    """Exécute une série de tests pour valider l'installation et la configuration."""
    utils.setup_logging(debug)
    logger.info("Lancement du test d'intégrité...")

    has_failed = False

    # 1. Check dependencies
    try:
        import torch
        import faster_whisper
        import transformers
        import yt_dlp
        logger.info("✅ Dépendances Python de base trouvées.")
        logger.info(f"    - PyTorch version: {torch.__version__}")
        logger.info(f"    - CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"    - Nom du GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        logger.error(f"❌ Dépendance manquante : {e.name}. Veuillez exécuter 'pip install -r requirements.txt'")
        has_failed = True

    # 2. Check FFmpeg
    try:
        utils.run_cmd(["ffmpeg", "-version"], check=True)
        logger.info("✅ FFmpeg est installé et accessible.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ FFmpeg n'est pas trouvé. Veuillez l'installer et vous assurer qu'il est dans le PATH.")
        has_failed = True

    # 3. Load ASR model and perform micro-inference
    try:
        logger.info("Chargement du modèle ASR pour test...")
        transcriber = Transcriber(model_name="tiny", compute_type="int8")
        # Create a dummy silent wav file for testing
        dummy_wav = Path("dummy_silent.wav")
        if not dummy_wav.exists():
            utils.run_cmd([
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                "-t", "1", "-q:a", "9", "-acodec", "pcm_s16le", str(dummy_wav)
            ])
        _ = transcriber.transcribe(str(dummy_wav))
        logger.info("✅ Modèle ASR chargé et test d'inférence réussi.")
        dummy_wav.unlink()
    except Exception as e:
        logger.error(f"❌ Échec du test du modèle ASR : {e}")
        has_failed = True

    # 4. Load LLM and perform micro-inference
    try:
        logger.info("Chargement du modèle LLM pour test (peut prendre du temps)...")
        # Use a very small model for health check to speed up the process
        translator = LLMTranslator(model_name="sshleifer/tiny-gpt2", quant="fp16")
        _ = translator._generate("test")
        logger.info("✅ Modèle LLM chargé et test d'inférence réussi.")
    except Exception as e:
        logger.error(f"❌ Échec du test du modèle LLM : {e}")
        has_failed = True

    # 5. Check YouTube connectivity
    try:
        logger.info("Test de la connectivité YouTube...")
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            ydl.extract_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ", download=False)
        logger.info("✅ Connexion à YouTube et récupération des métadonnées réussies.")
    except Exception as e:
        logger.error(f"❌ Échec de la connexion à YouTube : {e}")
        has_failed = True

    logger.info("-" * 20)
    if has_failed:
        logger.error("❌ Le test d'intégrité a échoué. Veuillez corriger les erreurs ci-dessus.")
        raise typer.Exit(code=1)
    else:
        logger.info("✅ Le test d'intégrité est réussi. L'environnement semble correctement configuré.")

if __name__ == "__main__":
    app()
