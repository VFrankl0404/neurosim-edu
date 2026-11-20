# NeuroSim-Edu: de los canales iónicos al aprendizaje automático

*Una ruta de simulación abierta y bilingüe (inglés / español) para enseñar neurociencia computacional a médicos y estudiantes de ciencias de la vida. Solo NumPy; cada línea a la vista.*
*[Read in English](README.md)*

| Cuaderno | Modelo | Qué se aprende | Vínculo clínico |
|---|---|---|---|
| `01_hodgkin_huxley` | Hodgkin-Huxley (1952) | corrientes iónicas, compuertas, excitabilidad de tipo II | farmacología de canales de Na⁺ y K⁺ |
| `02_integrate_and_fire` | Integración y disparo con fuga | umbral, reinicio, frecuencia exacta, error numérico | reobase |
| `03_izhikevich` | Izhikevich (2003) | cinco patrones corticales con cuatro parámetros | tipos celulares |
| `04_ei_network` | Red LIF de 800 E + 200 I | la inhibición convierte la sincronía en disparo irregular | crisis epilépticas, pérdida de interneuronas |
| `05_learning_machine` | Perceptrón 5-16-16-5 frente a regresión logística | clasificar neuronas simuladas; validación honesta, calibración | evaluar la IA clínica |
| `06_clinical_scenarios` | Hodgkin-Huxley perturbado | bloqueo de Na⁺ y K⁺, hiperpotasemia | lidocaína, 4-aminopiridina, electrolitos |
| `07_choosing_a_model` | todos | realismo frente a costo computacional | elegir un modelo para cada pregunta |

## Inicio rápido

```bash
pip install -r requirements.txt
jupyter notebook notebooks_es/              # Español
jupyter notebook notebooks/                 # English
python make_figures.py --lang es            # figuras del artículo con rótulos en español
python make_supplement_figures.py --lang es # figuras S1–S7 del cuaderno de práctica
```

El cuaderno de práctica en español está en `paper/Material_suplementario_S1_cuaderno_practica_ES.md`.

## Simulador en el navegador (sin instalación)

La carpeta `web/` contiene el simulador en el navegador (Material suplementario S3): archivos HTML únicos, en español y en inglés, que funcionan sin conexión en Microsoft Edge o Google Chrome. Basta con abrirlos con doble clic; no hay que instalar nada. Seis pestañas: Hodgkin-Huxley con escenarios clínicos (lidocaína, tetrodotoxina, 4-aminopiridina, hiperpotasemia, hipopotasemia, temperatura y calcio extracelular), LIF, Izhikevich, red E/I (con un preajuste de fármaco antiepiléptico), protocolos de Hodgkin-Huxley y un clasificador en vivo, con retos guiados, casos clínicos, glosario y hoja de trabajo imprimible.

## Licencia y cita

Código: MIT. Texto y figuras: CC BY 4.0. Cite el artículo y la versión archivada (ver `CITATION.cff`).

- Repositorio: [URL de GitHub: por añadir]
- Versión archivada: [DOI de Zenodo: por añadir]
