# Emanet-SRT : Outil de Transcription et Traduction Vidéo

Emanet-SRT est un outil en ligne de commande puissant conçu pour automatiser le processus de création de sous-titres. Il prend en charge le téléchargement de vidéos depuis des playlists YouTube, la transcription de l'audio en texte (turc), la traduction du texte en français, et la génération d'un fichier de sous-titres `.srt` final, propre et bien synchronisé.

L'outil est conçu pour fonctionner à 100% en local sur une machine équipée d'un GPU NVIDIA (optimisé pour RunPod), en utilisant des modèles de pointe open-source pour la transcription et la traduction.

## Fonctionnalités

- **Téléchargement depuis YouTube** : Gère les playlists complètes en téléchargeant uniquement l'audio pour un traitement efficace.
- **Transcription ASR de Haute Qualité** : Utilise un modèle de la famille `Voxtral` de Mistral AI pour une transcription rapide et précise avec horodatage au niveau du mot.
- **Traduction par LLM** : Emploie un grand modèle de langage (par défaut `Magistral-Small`) pour une traduction contextuelle et nuancée.
- **Synchronisation Intelligente** : Génère des sous-titres bien synchronisés en se basant sur l'horodatage des mots de l'audio original.
- **CLI Robuste et Conviviale** : Interface en ligne de commande basée sur `Typer` avec des commandes claires, des options documentées et des barres de progression pour chaque étape.
- **Test d'Intégrité** : Inclut une commande `health-check` pour valider l'environnement, les dépendances et le fonctionnement des modèles.
- **Haute Configurabilité** : La plupart des paramètres (modèles, quantification, etc.) sont configurables via le `Makefile` ou directement en ligne de commande.
- **Optimisé pour la Performance** : Inclut des optimisations telles que la quantification des modèles (`int4`, `int8`) et le traitement par lots (batching) pour maximiser l'utilisation du GPU.

---

## Prérequis

- **Système d'exploitation** : Linux (recommandé)
- **Python** : Version 3.10 ou supérieure
- **GPU** : Un GPU NVIDIA avec CUDA est fortement recommandé pour des performances acceptables. Le projet est optimisé pour un environnement de type RunPod B200.
- **FFmpeg** : Doit être installé et accessible dans le `PATH` de votre système. Utilisé pour la conversion audio.
  ```bash
  # Sur les systèmes basés sur Debian/Ubuntu
  sudo apt update && sudo apt install ffmpeg
  ```

---

## Installation

1.  **Clonez le dépôt :**
    ```bash
    git clone https://github.com/Grandgousier1/emanet-srt-latest.git
    cd emanet-srt-latest
    ```

2.  **Installez les dépendances Python :**
    Il est fortement recommandé d'utiliser un environnement virtuel.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    Utilisez la cible `install` du `Makefile` pour installer toutes les dépendances requises.
    ```bash
    make install
    ```
    Cette commande installera `PyTorch` pour CUDA, `transformers`, `bitsandbytes` et les autres bibliothèques nécessaires.

---

## Utilisation

Le `Makefile` est le point d'entrée recommandé pour utiliser l'application.

### 1. Test d'intégrité de l'environnement

Avant la première utilisation, il est conseillé de lancer un test de santé pour s'assurer que tout est correctement configuré.

```bash
make health-check
```

Cette commande vérifiera les dépendances, l'installation de FFmpeg, et tentera de charger des versions réduites des modèles ASR et LLM pour confirmer que l'inférence fonctionne.

### 2. Pré-télécharger les modèles

Pour éviter un long téléchargement lors de la première exécution, vous pouvez pré-télécharger les modèles dans le cache local.

```bash
make pull-models
```

### 3. Lancer le traitement

#### Sur une playlist YouTube

Modifiez la variable `PLAYLIST_URL` dans le `Makefile` ou passez-la en argument, puis exécutez :

```bash
make run-playlist
```

*Exemple avec une URL personnalisée :*
```bash
make run-playlist PLAYLIST_URL="https://www.youtube.com/playlist?list=VOTRE_LISTE_ID"
```

#### Sur des fichiers locaux

Placez vos fichiers vidéo ou audio dans un dossier (par exemple, `media/`). Modifiez la variable `LOCAL_FILES` dans le `Makefile` ou passez-la en argument.

```bash
# Assurez-vous de mettre les chemins entre guillemets s'ils contiennent des espaces
make run-local LOCAL_FILES="media/video1.mp4 media/épisode2.m4a"
```

Les fichiers SRT générés apparaîtront dans le dossier `output/`. Les fichiers de travail (audios téléchargés, WAVs) seront dans `workdir/`.

---

## Configuration

Vous pouvez facilement modifier le comportement de l'outil en ajustant les variables au début du `Makefile`.

- **`PLAYLIST_URL`**: L'URL de la playlist YouTube à traiter par défaut.
- **`LOCAL_FILES`**: Une chaîne de caractères contenant les chemins des fichiers locaux à traiter.
- **`ASR_MODEL`**: Le modèle de transcription `Voxtral` à utiliser (ex: `mistralai/Voxtral-Mini-3B-2507`).
- **`ASR_QUANT`**: La quantification à appliquer au modèle ASR (`int4`, `int8`). Laisser vide pour du float16/32.
- **`LLM_MODEL`**: Le modèle de langage à utiliser pour la traduction (ex: `mistralai/Magistral-Small-2507`).
- **`LLM_QUANT`**: La quantification à appliquer au LLM (`int4`, `int8`, `fp16`) pour optimiser l'usage mémoire.
- **`LLM_BATCH_SIZE`**: La taille du lot pour la traduction. Augmentez sur les GPU puissants pour plus de vitesse.

Toutes ces variables peuvent être surchargées directement depuis la ligne de commande :
```bash
make run-playlist ASR_MODEL="mistralai/Voxtral-Mini-..." ASR_QUANT="int8"
```

---

## Structure du Projet

```
.
├── emanet/              # Le code source du package Python
│   ├── __init__.py
│   ├── cli.py           # Définition de l'interface en ligne de commande (Typer)
│   ├── datastructures.py# Classes de données (Segment, Cue, etc.)
│   ├── downloader.py    # Gestion du téléchargement YouTube
│   ├── srt_builder.py   # Logique de construction du fichier SRT
│   ├── transcriber.py   # Module de transcription (ASR)
│   ├── translator.py    # Module de traduction (LLM)
│   └── utils.py         # Fonctions utilitaires
├── emanet_srt.py        # Point d'entrée principal de l'application
├── Makefile             # Point d'entrée pour les commandes utilisateur
├── requirements.txt     # Dépendances Python
└── README.md            # Ce fichier
```
