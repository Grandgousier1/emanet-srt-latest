# ==============================================================================
# Makefile pour le projet Emanet-SRT
# Conçu pour être exécuté directement sur un environnement pré-configuré (ex: RunPod).
# ==============================================================================

# --- Variables de configuration modifiables ---
PLAYLIST_URL ?= "https://www.youtube.com/playlist?list=PLjhol17mPBuP_QR6Bs-_ocNllsIX86_qx"
LOCAL_FILES ?= "" # Exemple: "media/video1.mp4 media/video2.mp3"

# Options ASR
ASR_MODEL ?= "large-v3"
ASR_COMPUTE_TYPE ?= "float16" # "float16", "int8_float16", "int8"
NO_ALIGN ?= false # 'true' pour désactiver, 'false' pour activer
NO_SPEECH_THRESHOLD ?= 0.6
LOGPROB_THRESHOLD ?= -1.25

# Options LLM
LLM_MODEL ?= "mistralai/Magistral-Small-2507"
LLM_QUANT ?= "int4" # "int4", "int8", "fp16"
LLM_WINDOW_SEC ?= 120
LLM_BATCH_SIZE ?= 8
STYLE ?= "neutral" # "neutral", "formal", "informal"

# Options SRT
MAX_CPS ?= 17
MAX_CHARS ?= 42

# --- Commandes ---
PYTHON := python3
PIP := pip3
CLI_ENTRY := $(PYTHON) emanet_srt.py

# Construction des flags pour la commande
# Si NO_ALIGN est 'true', on ajoute le flag --no-align
ALIGN_FLAG := $(if $(filter true,$(NO_ALIGN)),--no-align,)

PROCESS_ARGS = \
	--output-dir "output" \
	--work-dir "workdir" \
	--asr-model "$(ASR_MODEL)" \
	--asr-compute-type "$(ASR_COMPUTE_TYPE)" \
	$(ALIGN_FLAG) \
	--no-speech-threshold $(NO_SPEECH_THRESHOLD) \
	--logprob-threshold $(LOGPROB_THRESHOLD) \
	--llm-model "$(LLM_MODEL)" \
	--llm-quant "$(LLM_QUANT)" \
	--llm-window-sec $(LLM_WINDOW_SEC) \
	--llm-batch-size $(LLM_BATCH_SIZE) \
	--style "$(STYLE)" \
	--max-cps $(MAX_CPS) \
	--max-chars $(MAX_CHARS) \
	--save-debug-json \
	--debug

.PHONY: all help install run-playlist run-local health-check pull-models clean

all: help

help:
	@echo "Makefile pour Emanet-SRT"
	@echo ""
	@echo "Cibles disponibles:"
	@echo "  install         - Installe les dépendances Python via pip."
	@echo "  run-playlist    - Lance le traitement sur la PLAYLIST_URL par défaut."
	@echo "  run-local       - Lance le traitement sur les fichiers spécifiés dans LOCAL_FILES."
	@echo "  health-check    - Exécute les tests d'intégrité de l'environnement."
	@echo "  pull-models     - Pré-télécharge les modèles ASR et LLM dans le cache."
	@echo "  clean           - Supprime les dossiers de travail et de sortie."
	@echo ""
	@echo "Vous pouvez surcharger les variables de configuration, ex:"
	@echo "make run-local LOCAL_FILES='path/to/my/video.mp4' ASR_MODEL='medium'"

install:
	@echo "--- Installation des dépendances ---"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "--- Dépendances installées ---"

run-playlist:
	@echo "--- Lancement du traitement sur la playlist : $(PLAYLIST_URL) ---"
	$(CLI_ENTRY) process --playlist-url "$(PLAYLIST_URL)" $(PROCESS_ARGS)

run-local:
	@if [ -z "$(LOCAL_FILES)" ]; then echo "Erreur: La variable LOCAL_FILES est vide. Usage: make run-local LOCAL_FILES=\"/path/to/file1.mp4\""; exit 1; fi
	@echo "--- Lancement du traitement sur les fichiers locaux : $(LOCAL_FILES) ---"
	$(CLI_ENTRY) process --local-files $(LOCAL_FILES) $(PROCESS_ARGS)

health-check:
	@echo "--- Lancement du test d'intégrité ---"
	$(CLI_ENTRY) health-check --debug

pull-models:
	@echo "--- Pré-téléchargement des modèles (cela peut prendre du temps) ---"
	@echo "Modèle ASR: $(ASR_MODEL)"
	$(PYTHON) -c "from emanet.transcriber import Transcriber; Transcriber(model_name='$(ASR_MODEL)')"
	@echo "Modèle LLM: $(LLM_MODEL)"
	$(PYTHON) -c "from emanet.translator import LLMTranslator; LLMTranslator(model_name='$(LLM_MODEL)')"
	@echo "--- Modèles téléchargés ---"

clean:
	@echo "--- Nettoyage des dossiers de travail ---"
	rm -rf workdir output
	@echo "--- Nettoyage terminé ---"
