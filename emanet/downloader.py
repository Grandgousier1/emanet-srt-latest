import os
import logging
import subprocess
from typing import List, Optional, Dict

from tqdm import tqdm
from .utils import ensure_dir, run_cmd, to_srt_timestamp

logger = logging.getLogger("emanet")

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

class _TqdmYouTubeLogger:
    """A custom yt-dlp logger that integrates with a tqdm progress bar."""
    def __init__(self, pbar: tqdm):
        self.pbar = pbar

    def debug(self, msg):
        # yt-dlp requires a debug method, but we ignore its logs
        pass

    def info(self, msg):
        # We can log this if needed, e.g., for debugging yt-dlp behavior
        pass

    def warning(self, msg):
        logger.warning(f"[yt-dlp] {msg}")

    def error(self, msg):
        logger.error(f"[yt-dlp] {msg}")


def _pbar_hook(d, pbar: tqdm):
    """A yt-dlp progress hook that updates a tqdm progress bar."""
    if d['status'] == 'downloading':
        if pbar.total is None: # Set total size if not already set
            pbar.total = d.get('total_bytes') or d.get('total_bytes_estimate')
        pbar.update(d['downloaded_bytes'] - pbar.n)
    elif d['status'] == 'finished':
        if pbar.total is None:
            pbar.total = pbar.n
        else:
            pbar.n = pbar.total
        pbar.close()


class YouTubeAudioDownloader:
    """
    Handles downloading and preparing audio from YouTube playlists or local files.
    """
    def __init__(self, workdir: str, cookies: Optional[str] = None):
        """
        Initialise le downloader.

        Args:
            workdir (str): Dossier de travail pour stocker les fichiers audio.
            cookies (Optional[str]): Chemin vers un fichier de cookies pour yt-dlp.
        """
        if YoutubeDL is None:
            raise RuntimeError("yt-dlp n'est pas installé. Veuillez l'installer avec : pip install yt-dlp")
        self.workdir = workdir
        self.cookies = cookies
        ensure_dir(workdir)

    def download_playlist(self, url: str) -> List[Dict]:
        """
        Télécharge l'audio de toutes les vidéos d'une playlist YouTube.

        Args:
            url (str): L'URL de la playlist YouTube.

        Returns:
            List[Dict]: Une liste de dictionnaires, un pour chaque vidéo téléchargée avec succès.
        """
        base_ydl_opts = {
            "ignoreerrors": True,
            "format": "bestaudio/best",
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"},
            ],
            "noprogress": True,
            "quiet": True,
            "retries": 10,
            "fragment_retries": 10,
        }
        if self.cookies and os.path.isfile(self.cookies):
            base_ydl_opts["cookiefile"] = self.cookies

        logger.info("Analyse de la playlist YouTube...")
        with YoutubeDL(base_ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                if not info or not info.get("entries"):
                    logger.error("Impossible de récupérer les informations de la playlist. L'URL est-elle correcte ou la playlist est-elle privée/protégée ?")
                    return []
                entries = info.get("entries", [])
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse de la playlist : {e}")
                return []

        logger.info(f"{len(entries)} vidéo(s) trouvée(s) dans la playlist.")

        results = []
        for entry in tqdm(entries, desc="Téléchargement audio", unit="vidéo"):
            if not entry:
                continue

            vid_id = entry.get("id")
            playlist_idx = entry.get("playlist_index", len(results) + 1)
            title = entry.get("title") or f"video-{vid_id}"

            # NOTE: Le nom du fichier de sortie est construit à partir du template.
            # C'est généralement fiable, mais la méthode la plus robuste serait
            # de capturer le nom de fichier final via le hook de progression.
            # L'approche actuelle est un compromis pour la simplicité.
            outtmpl = os.path.join(self.workdir, f"{playlist_idx}-{vid_id}.%(ext)s")
            download_opts = {**base_ydl_opts, "outtmpl": outtmpl}

            # On ne retélécharge pas si le fichier existe déjà
            # Pour cela, on doit deviner le nom de fichier potentiel
            potential_path = os.path.join(self.workdir, f"{playlist_idx}-{vid_id}.m4a")
            if os.path.exists(potential_path):
                 logger.info(f"Fichier déjà téléchargé pour '{title}'. On passe.")
                 results.append({"id": vid_id, "index": playlist_idx, "title": title, "m4a": potential_path})
                 continue

            with tqdm(total=None, unit='B', unit_scale=True, desc=f"Vidéo {playlist_idx}", leave=False) as pbar:
                download_opts["progress_hooks"] = [lambda d, p=pbar: _pbar_hook(d, p)]
                download_opts["logger"] = _TqdmYouTubeLogger(pbar)

                with YoutubeDL(download_opts) as ydl_download:
                    try:
                        ydl_download.download([entry["webpage_url"]])
                    except Exception as e:
                        logger.error(f"Échec du téléchargement pour la vidéo '{title}': {e}")
                        continue

            # Recherche du fichier téléchargé, car l'extension peut varier
            downloaded_path = None
            for fname in os.listdir(self.workdir):
                if fname.startswith(f"{playlist_idx}-{vid_id}."):
                    downloaded_path = os.path.join(self.workdir, fname)
                    break

            if downloaded_path and os.path.exists(downloaded_path):
                results.append({"id": vid_id, "index": playlist_idx, "title": title, "m4a": downloaded_path})
            else:
                logger.warning(f"Fichier audio non trouvé pour la vidéo '{title}' après tentative de téléchargement.")

        logger.info(f"{len(results)}/{len(entries)} audios de la playlist sont prêts pour le traitement.")
        return results

    def convert_to_wav(self, in_path: str, out_wav: str, sr: int = 16000):
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
