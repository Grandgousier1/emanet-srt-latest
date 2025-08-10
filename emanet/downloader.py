import os
import logging
from typing import List, Optional, Dict

from tqdm import tqdm
from .utils import ensure_dir, run_cmd

logger = logging.getLogger("emanet")

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

class TqdmYouTubeLogger:
    def __init__(self, pbar: tqdm):
        self.pbar = pbar

    def debug(self, msg):
        # For compatibility with yt-dlp, we need a debug method.
        # We can log this if needed.
        pass

    def info(self, msg):
        # We can log this if needed.
        pass

    def warning(self, msg):
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)


def pbar_hook(d, pbar: tqdm):
    if d['status'] == 'downloading':
        pbar.total = d.get('total_bytes') or d.get('total_bytes_estimate')
        pbar.update(d['downloaded_bytes'] - pbar.n)
    elif d['status'] == 'finished':
        pbar.n = pbar.total or 0
        pbar.close()

class YouTubeAudioDownloader:
    def __init__(self, workdir: str, cookies: Optional[str] = None):
        if YoutubeDL is None:
            raise RuntimeError(
                "yt-dlp n'est pas installé. "
                "Veuillez l'installer avec : pip install yt-dlp"
            )
        self.workdir = workdir
        self.cookies = cookies
        ensure_dir(workdir)

    def download_playlist(self, url: str) -> List[Dict]:
        outtmpl = os.path.join(self.workdir, "%(playlist_index)s-%(id)s.%(ext)s")
        ydl_opts = {
            "ignoreerrors": True,
            "outtmpl": outtmpl,
            "format": "bestaudio/best",
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"},
            ],
            "noprogress": True, # We use our own progress bar
            "quiet": True,
            # Retry logic for transient errors
            "retries": 10,
            "fragment_retries": 10,
        }
        if self.cookies and os.path.isfile(self.cookies):
            ydl_opts["cookiefile"] = self.cookies

        logger.info("Analyse de la playlist YouTube...")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info or not info.get("entries"):
                logger.error("Impossible de récupérer les informations de la playlist. L'URL est-elle correcte ?")
                return []

            entries = info.get("entries", [])

        logger.info(f"{len(entries)} vidéo(s) trouvée(s) dans la playlist.")

        results = []
        for entry in tqdm(entries, desc="Téléchargement audio"):
            if not entry:
                continue

            vid_id = entry.get("id")
            playlist_idx = entry.get("playlist_index")
            title = entry.get("title") or f"video-{vid_id}"

            # We need to call download for each video to get progress
            with tqdm(total=100, unit='B', unit_scale=True, desc=f"Vidéo {playlist_idx}", leave=False) as pbar:
                ydl_opts["progress_hooks"] = [lambda d: pbar_hook(d, pbar)]
                ydl_opts["logger"] = TqdmYouTubeLogger(pbar)

                with YoutubeDL(ydl_opts) as ydl_download:
                    try:
                        ydl_download.download([entry["webpage_url"]])
                    except Exception as e:
                        logger.error(f"Échec du téléchargement pour la vidéo {title}: {e}")
                        continue

            # Find the downloaded file
            m4a_path = os.path.join(self.workdir, f"{playlist_idx}-{vid_id}.m4a")
            if not os.path.exists(m4a_path):
                 # yt-dlp might save with a different extension
                for fname in os.listdir(self.workdir):
                    if fname.startswith(f"{playlist_idx}-{vid_id}."):
                        m4a_path = os.path.join(self.workdir, fname)
                        break

            if os.path.exists(m4a_path):
                results.append({"id": vid_id, "index": playlist_idx, "title": title, "m4a": m4a_path})
            else:
                logger.warning(f"Fichier audio non trouvé pour la vidéo {title} après téléchargement.")

        logger.info(f"{len(results)}/{len(entries)} audios téléchargés avec succès.")
        return results

    def convert_to_wav(self, in_path: str, out_wav: str, sr: int = 16000):
        ensure_dir(os.path.dirname(out_wav))
        if os.path.exists(out_wav):
            logger.debug(f"Le fichier WAV existe déjà : {out_wav}")
            return

        logger.info(f"Conversion de {os.path.basename(in_path)} en WAV 16kHz mono...")
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", str(sr), "-vn", out_wav]
        try:
            run_cmd(cmd)
            logger.info("Conversion WAV terminée.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de la conversion WAV : {e.stderr}")
            raise
