#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Emanet-SRT: Outil de transcription et de traduction de vidéos en sous-titres.
Ce script est le point d'entrée principal de l'application.
Il lance l'interface en ligne de commande (CLI) définie dans le module emanet.cli.
"""

from emanet.cli import app

if __name__ == "__main__":
    app()
