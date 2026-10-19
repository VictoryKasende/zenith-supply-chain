"""Expériences de révision de l'article (validité expérimentale et reproductibilité).

Toutes les sorties vont dans ``paper_results/``. Point d'entrée unique :
``python -m experiments.run_all``.
"""
import os
import sys

# Un seul thread BLAS / OpenMP, fixé avant tout import numérique : les
# résultats ne dépendent ni du nombre de cœurs ni de l'ordonnancement des
# processus parallèles (LightGBM et l'algèbre linéaire somment dans un ordre
# fixe).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

# Console Windows (cp1252) : les sorties contiennent ≥, →, etc.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
