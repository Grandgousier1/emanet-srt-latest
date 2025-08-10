import re
import logging
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .datastructures import Segment

logger = logging.getLogger("emanet")

class LLMTranslator:
    """
    Gère la traduction de texte en utilisant un grand modèle de langage (LLM) local.
    Optimisé pour le traitement par lots (batching) afin de maximiser les performances GPU.
    """
    def __init__(
        self,
        model_name: str = "mistralai/Magistral-Small-2507",
        quant: str = "int4",  # "int4", "int8", "fp16"
        window_sec: int = 120,
        style: str = "neutral",  # "neutral", "formal", "informal"
        max_new_tokens: int = 2048,
        batch_size: int = 8,
        source_lang: str = "turc",
        target_lang: str = "français",
    ):
        """
        Initialise le traducteur LLM.

        Args:
            model_name (str): Nom du modèle Hugging Face à utiliser.
            quant (str): Type de quantification ("int4", "int8", "fp16").
            window_sec (int): Taille de la fenêtre de contexte en secondes.
            style (str): Style de traduction ("neutral", "formal", "informal").
            max_new_tokens (int): Nombre maximum de tokens à générer.
            batch_size (int): Taille du lot pour l'inférence.
            source_lang (str): Langue source pour la traduction.
            target_lang (str): Langue cible pour la traduction.
        """
        self.model_name = model_name
        self.quant = quant
        self.window_sec = window_sec
        self.style = style
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.source_lang = source_lang
        self.target_lang = target_lang

        logger.info(f"Chargement du LLM traducteur: {model_name} (quant: {quant}, batch_size: {batch_size})...")

        load_kwargs = {"device_map": "auto"}
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        if quant == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype
            )
        elif quant == "int8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = dtype

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.warning("Le pad_token n'était pas défini. Utilisation de eos_token comme pad_token.")

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            self.has_chat_template = bool(self.tokenizer.chat_template)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
            raise

    def _system_prompt(self) -> str:
        """Génère le prompt système en fonction du style demandé."""
        style_map = {
            "neutral": "un registre de langue naturel et neutre",
            "formal": "un registre soutenu, en utilisant le vouvoiement",
            "informal": "un registre familier, en utilisant le tutoiement",
        }
        style_txt = style_map.get(self.style, style_map["neutral"])
        return (
            f"Tu es un traducteur expert spécialisé dans la traduction du {self.source_lang} vers le {self.target_lang}. "
            "Ta traduction doit être fidèle au sens original, idiomatique et fluide. "
            f"Adopte {style_txt}. Ne fournis AUCUNE note, explication ou commentaire. "
            "Réponds uniquement avec le texte traduit. Conserve les sauts de ligne du texte source."
        )

    def _build_prompts(self, tr_texts: List[str]) -> List[str]:
        """Construit une liste de prompts complets pour un lot de textes."""
        prompts = []
        for tr_text in tr_texts:
            if self.has_chat_template:
                messages = [{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": tr_text}]
                prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            else:
                # Fallback pour les modèles sans chat template. Le format peut varier.
                prompts.append(f"<|system|>\n{self._system_prompt()}</s>\n<|user|>\n{tr_text}</s>\n<|assistant|>\n")
        return prompts

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Génère des traductions pour un lot de prompts."""
        # Tokenize le lot de prompts
        self.tokenizer.padding_side = "left" # Le padding à gauche est crucial pour la génération
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Décode seulement les tokens générés
        input_len = inputs['input_ids'].shape[1]
        new_tokens = outputs[:, input_len:]
        decoded_texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [text.strip() for text in decoded_texts]

    def translate_segments(self, segments: List[Segment]) -> List[str]:
        """
        Traduit une liste de segments en utilisant le LLM, avec traitement par lots.

        Args:
            segments (List[Segment]): La liste des segments transcrits.

        Returns:
            List[str]: Une liste des blocs de texte traduits.
        """
        if not segments:
            return []

        # 1. Grouper les segments en fenêtres de contexte
        windows = []
        current_window = []
        if segments:
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

        # 2. Préparer les textes sources pour chaque fenêtre
        source_texts = ["\n".join([s.text for s in win if s.text.strip()]) for win in windows]
        source_texts = [text for text in source_texts if text] # Filtrer les fenêtres vides

        logger.info(f"Traduction avec le LLM en {len(source_texts)} fenêtre(s) de contexte (~{self.window_sec}s chacune).")

        # 3. Traiter par lots (batch)
        translated_blocks = []
        pbar_desc = f"Traduction LLM (lot de {self.batch_size})"
        for i in tqdm(range(0, len(source_texts), self.batch_size), desc=pbar_desc):
            batch_texts = source_texts[i:i + self.batch_size]
            batch_prompts = self._build_prompts(batch_texts)
            translated_batch = self._generate_batch(batch_prompts)
            translated_blocks.extend(translated_batch)

        return translated_blocks
