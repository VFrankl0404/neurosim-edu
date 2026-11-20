# NeuroSim-Edu: from ion channels to machine learning

*An open, bilingual (English / Spanish) simulation pathway for teaching computational neuroscience to physicians and life-science students. NumPy only; every line visible.*
*[Leer en español](README_es.md)*

| Notebook | Model | What you learn | Clinical link |
|---|---|---|---|
| `01_hodgkin_huxley` | Hodgkin-Huxley (1952) | ionic currents, gating, type II excitability | Na⁺/K⁺ channel pharmacology |
| `02_integrate_and_fire` | Leaky integrate-and-fire | threshold, reset, closed-form rate, numerical error | rheobase |
| `03_izhikevich` | Izhikevich (2003) | five cortical firing patterns from four parameters | cell types |
| `04_ei_network` | 800 E + 200 I LIF network | inhibition turns synchrony into irregular firing | seizures, interneuron loss |
| `05_learning_machine` | 5-16-16-5 perceptron vs logistic regression | classifying simulated neurons; honest validation, calibration | appraising clinical AI |
| `06_clinical_scenarios` | perturbed Hodgkin-Huxley | Na⁺ block, K⁺ block, hyperkalaemia | lidocaine, 4-aminopyridine, electrolytes |
| `07_choosing_a_model` | all | realism versus computational cost | choosing a model for a question |

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/                 # English
jupyter notebook notebooks_es/              # Español
python make_figures.py                      # article figures + results/results.json (2–4 min)
python make_figures.py --lang es            # Spanish-labelled figures
python make_supplement_figures.py           # workbook figures S1–S7 (add --lang es for Spanish)
python build_notebooks.py                   # rewrite both notebook sets from one source
```

## Browser simulator (no installation)

The `web/` folder contains the browser simulator (Supplementary File S3): single HTML files, in English and Spanish, that run offline in Microsoft Edge or Google Chrome. Double-click to open; nothing to install. Six tabs: Hodgkin-Huxley with clinical scenarios (lidocaine, tetrodotoxin, 4-aminopyridine, hyperkalaemia, hypokalaemia, temperature, extracellular calcium), LIF, Izhikevich, E/I network (including an antiseizure-drug preset), Hodgkin-Huxley protocols and a live classifier, with guided challenges, clinical cases, a glossary and a printable worksheet.

## Repository layout

```
neurosim/                Python package (NumPy only, ~690 lines with docstrings)
  hh.py                  Hodgkin-Huxley model, spike detection, settling
  point.py               LIF and Izhikevich (scalar and vectorised)
  network.py             excitatory-inhibitory LIF network and its statistics
  classifier.py          data set, features, perceptron, evaluation statistics
  clinic.py              sodium/potassium block, extracellular potassium, f-I curves
notebooks/, notebooks_es/        seven teaching notebooks with exercises
figures/, figures_es/            article figures (PNG 600 dpi + PDF)
figures_supp/, figures_supp_es/  workbook figures S1–S7
results/                 every number in the article (results.json, supplement_results.json)
paper/                   manuscript, Supplementary Files S1 (EN/ES) and S2
submission/              title page, cover letter, declarations
web/                     browser simulator (single-file HTML, English and Spanish)
```

## License and citation

Code: MIT. Text and figures: CC BY 4.0. Please cite the article and the archived release (see `CITATION.cff`).

- Repository: [GitHub URL — to be added]
- Archived release: [Zenodo DOI — to be added]
