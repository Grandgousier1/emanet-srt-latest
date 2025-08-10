import re
import logging
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .datastructures import Segment

logger = logging.getLogger("emanet")

class LLMTranslator:
    """
    Handles text translation using a local Large Language Model.
    """
    def __init__(
        self,
        model_name: str = "mistralai/Magistral-Small-2507",
        quant: str = "int4",  # "int4", "int8", "fp16"
        window_sec: int = 120,
        style: str = "neutral",  # "neutral", "formal", "informal"
        max_new_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.quant = quant
        self.window_sec = window_sec
        self.style = style
        self.max_new_tokens = max_new_tokens

        logger.info(f"Chargement du LLM traducteur: {model_name} (quant: {quant})...")

        load_kwargs = {"device_map": "auto"}
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        if quant == "int4":
            try:
                # For new transformers versions with bitsandbytes
                load_kwargs.update(dict(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype
                ))
            except Exception:
                logger.warning("Quantification 4-bit (bitsandbytes) indisponible, fallback sur FP16/BF16.")
                load_kwargs["torch_dtype"] = dtype
        elif quant == "int8":
            try:
                load_kwargs.update(dict(load_in_8bit=True))
            except Exception:
                logger.warning("Quantification 8-bit indisponible, fallback sur FP16/BF16.")
                load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = dtype

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            self.has_chat_template = bool(self.tokenizer.chat_template)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
            raise

    def _system_prompt(self) -> str:
        style_map = {
            "neutral": "un registre de langue naturel et neutre",
            "formal": "un registre soutenu, en utilisant le vouvoiement",
            "informal": "un registre familier, en utilisant le tutoiement",
        }
        style_txt = style_map.get(self.style, style_map["neutral"])
        return (
            "Tu es un traducteur expert spécialisé dans la traduction du turc vers le français. "
            "Ta traduction doit être fidèle au sens original, idiomatique et fluide. "
            f"Adopte {style_txt}. Ne fournis AUCUNE note, explication ou commentaire. "
            "Réponds uniquement avec le texte traduit. Conserve les sauts de ligne du texte source."
        )

    def _build_prompt(self, tr_text: str) -> str:
        if self.has_chat_template:
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": tr_text},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Impossible d'appliquer le chat template, fallback sur le prompt manuel. Erreur: {e}")
                # Fallback if template is broken or not fully supported
                return f"<|system|>\n{self._system_prompt()}</s>\n<|user|>\n{tr_text}</s>\n<|assistant|>\n"
        else:
            # Manual prompt for models without a chat template
            return f"<|system|>\n{self._system_prompt()}</s>\n<|user|>\n{tr_text}</s>\n<|assistant|>\n"

    def _generate(self, prompt: str) -> str:
        tok = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **tok,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = out_ids[0, tok['input_ids'].shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text

    def translate_segments(self, segments: List[Segment]) -> List[str]:
        if not segments:
            return []

        # Group segments into windows to provide context
        windows: List[List[Segment]] = []
        current_window: List[Segment] = []
        current_start_time = segments[0].start

        for s in segments:
            if not current_window:
                current_window = [s]
                current_start_time = s.start
                continue

            if (s.end - current_start_time) <= self.window_sec:
                current_window.append(s)
            else:
                windows.append(current_window)
                current_window = [s]
                current_start_time = s.start
        if current_window:
            windows.append(current_window)

        logger.info(f"Traduction avec le LLM en {len(windows)} fenêtre(s) de contexte (~{self.window_sec}s chacune).")

        translated_blocks: List[str] = []
        for win in tqdm(windows, desc="Traduction LLM (TR→FR)"):
            tr_text = "\n".join([s.text for s in win if s.text.strip()])
            if not tr_text.strip():
                continue

            prompt = self._build_prompt(tr_text)
            fr_text = self._generate(prompt)
            translated_blocks.append(fr_text)

        return translated_blocks
