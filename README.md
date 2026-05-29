# Multidimensional Financial Vulnerability & Cyber-Fraud Risk Pipelines in Italy
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end micro-analytical framework designed to evaluate financial vulnerability, survey item non-disclosure, and cyber-fraud susceptibility in the Italian adult population. Utilizing the Bank of Italy's **IACOFI 2023** dataset, this repository implements a two-stage unsupervised manifold clustering pipeline, an explainable AI (XAI) framework for informative missingness decoding, and an operationally optimized cost-sensitive predictive system.

---

## Technical Architecture Overview

The repository is structured around three primary analytical blocks engineered to handle highly sparse, zero-inflated, and non-linearly distributed survey microdata:

1. **Topological Unsupervised Profiling (RQ1a):** Combines a Self-Organizing Map (SOM) lattice for high-dimensional geometric compression with a secondary $K$-means meta-clustering execution ($K=4$) to isolate continuous consumer behavioral archetypes.
2. **Informative Non-Disclosure Analytics (RQ1b):** Transformed income non-response patterns into an explicit behavioral signal. Modeled via a multivariate Random Forest classifier coupled with SHAP (SHapley Additive exPlanations) value decompositions to segment reticent respondents into distinct socioeconomic archetypes.
3. **Cost-Sensitive Cyber-Fraud Pipeline (RQ2):** Implements an operationally viable classification model mapping fraud susceptibility. Embeds an asymmetric business cost matrix ($FN:FP = 5:1$) into an algorithmic loss function to optimize banking risk capital burden and overcome data-scarcity blind spots.

---

## Dataset & Feature Engineering

The codebase leverages the **Indagine sulle Competenze Finanziarie degli Adulti in Italia (IACOFI 2023)** microdata compiled by Banca d'Italia. 

### Preprocessing Pipelines
* **Conditional Routing Restoration:** Skip-logic structural NaNs (e.g., product ownership filters or digital exclusions) are conditionally mapped to logical baseline vectors to maintain matrix dimensionality without injecting bias.
* **Informative Missingness Mappings:** Missing values in sensitive risk aversion metrics and household income are preserved as explicit tracking features rather than dropped or deterministically imputed.
* **Composite Indicator Construction:** Raw survey matrices are engineered into multi-choice continuous behavioral and psychological dimensions, including an objective-subjective **Financial Literacy Gap** scale, a *Saving Sophistication Level* scale, and directional *Attitudinal Consolidation indicators* (reverse-coded for valence alignment).

---

To explore the analysis and run the code locally, clone this repository using the following command:
```bash

git clone https://github.com/aronetommaso/DS_Lab.git

cd DS_Lab
