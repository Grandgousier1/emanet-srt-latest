import os
import logging
import subprocess
import time
import random
from typing import List, Optional, Dict, Tuple

from tqdm import tqdm
from .utils import ensure_dir, run_cmd

logger = logging.getLogger("emanet")

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

# User-Agent commun pour paraître moins comme un script
COMMON_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"

class _TqdmYouTubeLogger:
    """Un logger yt-dlp personnalisé qui s'intègre à une barre de progression tqdm."""
    def __init__(self, pbar: tqdm):
        self.pbar = pbar

    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): logger.warning(f"[yt-dlp] {msg}")
    def error(self, msg): logger.error(f"[yt-dlp] {msg}")

def _create_pbar_hook(pbar: tqdm, state: Dict):
    """Crée un hook de progression pour yt-dlp qui met à jour une barre tqdm et capture le nom de fichier."""
    def hook(d):
        if d['status'] == 'downloading':
            if pbar.total is None:
                pbar.total = d.get('total_bytes') or d.get('total_bytes_estimate')
            pbar.update(d['downloaded_bytes'] - pbar.n)
        elif d['status'] == 'finished':
            if pbar.total is None: pbar.total = pbar.n
            else: pbar.n = pbar.total
            state['filename'] = d.get('filename')
            pbar.close()
    return hook

class YouTubeAudioDownloader:
    """
    Gère le téléchargement et la préparation de l'audio depuis YouTube.
    """
    def __init__(
        self,
        workdir: str,
        cookies: Optional[str] = None,
        user_agent: Optional[str] = None,
        download_delay: Tuple[float, float] = (1.0, 5.0)
    ):
        """
        Initialise le downloader.
        Args:
            workdir (str): Dossier de travail pour stocker les fichiers audio.
            cookies (Optional[str]): Chemin vers un fichier de cookies pour yt-dlp.
            user_agent (str): User-Agent à utiliser pour les requêtes.
            download_delay (Tuple[float, float]): Délai aléatoire (min, max) en secondes entre les téléchargements.
        """
        if YoutubeDL is None:
            raise RuntimeError("yt-dlp n'est pas installé. Veuillez l'installer avec : pip install yt-dlp")
        self.workdir = workdir
        self.cookies = cookies
        self.user_agent = user_agent or COMMON_USER_AGENT
        self.download_delay = download_delay
        ensure_dir(workdir)

    def download_playlist(self, url: str) -> List[Dict]:
        """
        Télécharge l'audio de toutes les vidéos d'une playlist YouTube.
        """
        base_ydl_opts = {
            "ignoreerrors": True,
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"}],
            "noprogress": True, "quiet": True, "retries": 10, "fragment_retries": 10,
            "http_headers": {"User-Agent": self.user_agent},
        }
        if self.cookies and os.path.isfile(self.cookies):
            base_ydl_opts["cookiefile"] = self.cookies

        logger.info("Analyse de la playlist YouTube...")
        try:
            with YoutubeDL(base_ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info or not info.get("entries"):
                    logger.error("Impossible de récupérer les infos de la playlist. URL incorrecte ou playlist privée/protégée?")
                    return []
                entries = info.get("entries", [])
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la playlist : {e}")
            return []

        logger.info(f"{len(entries)} vidéo(s) trouvée(s) dans la playlist.")
        results = []
        for entry in tqdm(entries, desc="Téléchargement audio", unit="vidéo"):
            if not entry: continue

            vid_id = entry.get("id")
            playlist_idx = entry.get("playlist_index", len(results) + 1)
            title = entry.get("title") or f"video-{vid_id}"

            # Nom de fichier de sortie prévisible
            outtmpl = os.path.join(self.workdir, f"{playlist_idx:03d}-{vid_id}.%(ext)s")
            potential_path = os.path.join(self.workdir, f"{playlist_idx:03d}-{vid_id}.m4a")

            if os.path.exists(potential_path):
                 logger.info(f"Fichier déjà téléchargé pour '{title}'. On passe.")
                 results.append({"id": vid_id, "index": playlist_idx, "title": title, "m4a": potential_path})
                 continue

            # Appliquer un délai pour un comportement plus humain
            delay = random.uniform(*self.download_delay)
            logger.debug(f"Pause de {delay:.1f}s avant le prochain téléchargement.")
            time.sleep(delay)

            hook_state = {'filename': None}
            download_opts = {**base_ydl_opts, "outtmpl": outtmpl}

            with tqdm(total=None, unit='B', unit_scale=True, desc=f"Vidéo {playlist_idx}", leave=False) as pbar:
                download_opts["progress_hooks"] = [_create_pbar_hook(pbar, hook_state)]
                download_opts["logger"] = _TqdmYouTubeLogger(pbar)

                with YoutubeDL(download_opts) as ydl_download:
                    try:
                        # On utilise `extract_info` avec download=True pour s'assurer que le hook reçoit bien les infos
                        ydl_download.extract_info(entry["webpage_url"], download=True)
                    except Exception as e:
                        logger.error(f"Échec du téléchargement pour la vidéo '{title}': {e}")
                        continue

            downloaded_path = hook_state.get('filename')
            if downloaded_path and os.path.exists(downloaded_path):
                results.append({"id": vid_id, "index": playlist_idx, "title": title, "m4a": downloaded_path})
            else:
                # Fallback: si le hook a échoué mais que le fichier existe quand même
                if os.path.exists(potential_path):
                    logger.warning(f"Le hook n'a pas retourné de chemin, mais le fichier a été trouvé à l'emplacement attendu pour '{title}'.")
                    results.append({"id": vid_id, "index": playlist_idx, "title": title, "m4a": potential_path})
                else:
                    logger.warning(f"Fichier audio non trouvé pour '{title}' après tentative de téléchargement.")

        logger.info(f"{len(results)}/{len(entries)} audios de la playlist sont prêts pour le traitement.")
        return results

    @staticmethod
    def convert_to_wav(in_path: str, out_wav: str, sr: int = 16000):
        """
        Convertit un fichier audio en format WAV mono 16kHz avec FFmpeg.

        Args:
            in_path (str): Chemin du fichier audio source.
            out_wav (str): Chemin du fichier WAV de destination.
            sr (int): Taux d'échantillonnage cible.
        """
        ensure_dir(os.path.dirname(out_wav))
        if os.path.exists(out_wav):
            logger.debug(f"Le fichier WAV existe déjà, pas de reconversion : {out_wav}")
            return

        logger.info(f"Conversion de '{os.path.basename(in_path)}' en WAV 16kHz mono...")
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", str(sr), "-vn", out_wav]
        try:
            run_cmd(cmd)
            logger.info("Conversion WAV terminée.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur FFmpeg lors de la conversion WAV : {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("La commande 'ffmpeg' n'a pas été trouvée. Assurez-vous que FFmpeg est installé et dans le PATH.")
            raise
