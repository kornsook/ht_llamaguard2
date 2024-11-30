**LLaMaGurad 2 for Human Trafficking**

There are three files in this repository:
1) add_llamaguard_label.py: This file runs the LLaMaGurad 2 on a dataset (e.g., HTRP, Switter and etc.) and we use the predictions as LLaMaGuard labels.
2) evaluate_htrp.py: This file evaluates the LLaMaGuard labels against the labels (groud truth).
3) compare.py: This file compares the LLaMaGuard labels between two datasets (non-NERed and NERed) to see how much NER changes the LLaMaGuard 2's predictions.
