IMAGE := emanet-srt:latest
PLAYLIST := https://youtube.com/playlist?list=PLjhol17mPBuP_QR6Bs-_ocNllsIX86_qx
ASR_MODEL := large-v3
ASR_COMPUTE := float16           # ou int8_float16 pour économiser la VRAM
ALIGN := 1                       # 1 = align WhisperX, 0 = sans
LLM_MODEL := Qwen/Qwen2.5-7B-Instruct
LLM_QUANT := int4
LLM_WINDOW := 120
STYLE := neutral                 # neutral | formal | informal

DOCKER_RUN := docker run --gpus all --rm -it \
	-v $(PWD)/output:/app/output \
	-v $(PWD)/workdir:/app/workdir \
	-v $(PWD)/hf_cache:/app/hf_cache

build:
	docker build -t $(IMAGE) -f Dockerfile .

bash:
	$(DOCKER_RUN) --entrypoint bash $(IMAGE)

run-playlist:
	$(DOCKER_RUN) $(IMAGE) \
	  --playlist-url "$(PLAYLIST)" \
	  --asr-model $(ASR_MODEL) \
	  --asr-compute-type $(ASR_COMPUTE) \
	  $(if $(filter $(ALIGN),0),--no-align,) \
	  --llm-model "$(LLM_MODEL)" \
	  --llm-quant $(LLM_QUANT) \
	  --llm-window-sec $(LLM_WINDOW) \
	  --style $(STYLE) \
	  --save-debug

run-local:
	@if [ -z "$(FILES)" ]; then echo "Usage: make run-local FILES='/media/ep1.mp4 /media/ep2.m4a'"; exit 1; fi
	$(DOCKER_RUN) -v $(PWD)/media:/media $(IMAGE) \
	  --local-files $(FILES) \
	  --asr-model $(ASR_MODEL) \
	  --asr-compute-type $(ASR_COMPUTE) \
	  $(if $(filter $(ALIGN),0),--no-align,) \
	  --llm-model "$(LLM_MODEL)" \
	  --llm-quant $(LLM_QUANT) \
	  --llm-window-sec $(LLM_WINDOW) \
	  --style $(STYLE) \
	  --save-debug

pull-models:
	# Pré-télécharge les modèles dans le cache HF
	$(DOCKER_RUN) --entrypoint python3 $(IMAGE) - << 'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
m = "Qwen/Qwen2.5-7B-Instruct"
print("Pulling", m)
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True)
print("OK")
PY

clean:
	rm -rf output workdir hf_cache
