"""Write the seven teaching notebooks in English (notebooks/) and Spanish
(notebooks_es/) from one source, so both versions always run identical code.

    python build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

SETUP = """import sys, inspect
sys.path.insert(0, '..')            # make the neurosim package importable
import numpy as np
import matplotlib.pyplot as plt
import neurosim as ns
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})"""


def md(en, es):
    return ("md", {"en": en, "es": es})


def code(src):
    return ("code", src)


NB = {}

# --------------------------------------------------------------------- 01
NB["01_hodgkin_huxley"] = [
    md("""# Module 1 · Where does the action potential come from? (Hodgkin-Huxley)

**Clinical hook.** Local anaesthetics, many antiseizure drugs and tetrodotoxin poisoning all act on the same molecule: the voltage-gated sodium channel. To reason about them you need a picture of how Na⁺ and K⁺ currents build a spike.

**The model.** The membrane is a capacitor in parallel with three conductances:

$$C\\,\\frac{dV}{dt} = I_{stim} - \\bar g_{Na}\\,m^3h\\,(V-E_{Na}) - \\bar g_K\\,n^4\\,(V-E_K) - g_L\\,(V-E_L)$$

and every gate $x\\in\\{m,h,n\\}$ follows $dx/dt = \\alpha_x(V)(1-x) - \\beta_x(V)\\,x$.

**By the end you can** (1) say which ion carries the upstroke and which the repolarisation, (2) read gate plots, (3) recognise type II excitability in an f-I curve.""",
       """# Módulo 1 · ¿De dónde sale el potencial de acción? (Hodgkin-Huxley)

**Gancho clínico.** Los anestésicos locales, muchos antiepilépticos y la intoxicación por tetrodotoxina actúan sobre la misma molécula: el canal de sodio dependiente de voltaje. Para razonar sobre ellos hace falta una imagen de cómo las corrientes de Na⁺ y K⁺ construyen un disparo.

**El modelo.** La membrana es un condensador en paralelo con tres conductancias:

$$C\\,\\frac{dV}{dt} = I_{estím} - \\bar g_{Na}\\,m^3h\\,(V-E_{Na}) - \\bar g_K\\,n^4\\,(V-E_K) - g_L\\,(V-E_L)$$

y cada compuerta $x\\in\\{m,h,n\\}$ sigue $dx/dt = \\alpha_x(V)(1-x) - \\beta_x(V)\\,x$.

**Al terminar podrás** (1) decir qué ion produce la subida y cuál la repolarización, (2) leer gráficas de compuertas y (3) reconocer la excitabilidad de tipo II en una curva f-I."""),
    code(SETUP),
    md("Read the simulator next to the equation: each line inside the loop is one term.",
       "Lee el simulador junto a la ecuación: cada línea dentro del bucle es un término."),
    code("print(inspect.getsource(ns.simulate_hh))"),
    md("## 1.1 A current pulse", "## 1.1 Un pulso de corriente"),
    code("""pulse = lambda t: 10.0 if 10 <= t < 60 else 0.0     # µA/cm² between 10 and 60 ms
r = ns.simulate_hh(pulse, duration=80)
fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
ax[0].plot(r['t'], r['V'], color='#882255'); ax[0].set_ylabel('V (mV)')
for g, ls in zip('mhn', ['-', '--', '-.']): ax[1].plot(r['t'], r[g], ls, label=g)
ax[1].set_ylabel('open fraction'); ax[1].legend(ncol=3)
ax[2].plot(r['t'], r['I_Na'], label='I_Na'); ax[2].plot(r['t'], r['I_K'], label='I_K')
ax[2].set_ylabel('µA/cm²'); ax[2].set_xlabel('time (ms)'); ax[2].legend(); plt.show()"""),
    md("## 1.2 The frequency-current (f-I) curve", "## 1.2 La curva frecuencia-corriente (f-I)"),
    code("""currents = np.arange(0, 20.1, 0.5)
rates = [ns.hh_rate(I) for I in currents]
plt.plot(currents, rates, 'D-', ms=3, color='#882255')
plt.xlabel('current (µA/cm²)'); plt.ylabel('rate (Hz)'); plt.show()"""),
    md("""## Exercises
1. **TTX.** Block all sodium channels with `params={'gNa': 0}`. What happens to the pulse response, and why?
2. **Potassium.** Halve `gK`. How do spike width and rate change?
3. **Threshold for a brief pulse.** Find the smallest amplitude of a 1 ms pulse that evokes one spike (hint: try 5 to 10 µA/cm²).
4. **Type II.** Why does the f-I curve jump from 0 to about 50 Hz instead of rising smoothly from 0?
5. **Temperature (challenge).** Gate kinetics speed up about threefold for each 10 °C (Q10 ≈ 3). Multiply all rates by 3 inside a copy of `gate_rates` and describe the effect on spike shape.""",
       """## Ejercicios
1. **TTX.** Bloquea todos los canales de sodio con `params={'gNa': 0}`. ¿Qué le pasa a la respuesta al pulso y por qué?
2. **Potasio.** Reduce `gK` a la mitad. ¿Cómo cambian el ancho del disparo y la frecuencia?
3. **Umbral para un pulso breve.** Encuentra la amplitud mínima de un pulso de 1 ms que produce un disparo (pista: prueba entre 5 y 10 µA/cm²).
4. **Tipo II.** ¿Por qué la curva f-I salta de 0 a unos 50 Hz en lugar de subir suavemente desde 0?
5. **Temperatura (reto).** La cinética de las compuertas se acelera unas tres veces por cada 10 °C (Q10 ≈ 3). Multiplica todas las velocidades por 3 en una copia de `gate_rates` y describe el efecto en la forma del disparo."""),
]

# --------------------------------------------------------------------- 02
NB["02_integrate_and_fire"] = [
    md("""# Module 2 · Keeping only threshold and reset (LIF)

The leaky integrate-and-fire neuron keeps the passive membrane and replaces the spike by a rule:

$$\\tau_m\\frac{dV}{dt} = -(V-E_L) + R_m I, \\qquad V\\ge V_{th} \\Rightarrow \\text{spike},\\ V\\leftarrow V_{reset}$$

Because the equation is linear its firing rate has an exact formula, which lets us **test the numerical method** against the truth:

$$f(I) = \\Big[t_{ref} + \\tau_m\\ln\\frac{R_mI}{R_mI-(V_{th}-E_L)}\\Big]^{-1}$$""",
       """# Módulo 2 · Quedarse solo con umbral y reinicio (LIF)

La neurona de integración y disparo con fuga conserva la membrana pasiva y reemplaza el disparo por una regla:

$$\\tau_m\\frac{dV}{dt} = -(V-E_L) + R_m I, \\qquad V\\ge V_{th} \\Rightarrow \\text{disparo},\\ V\\leftarrow V_{reset}$$

Como la ecuación es lineal, su frecuencia de disparo tiene una fórmula exacta, que nos permite **poner a prueba el método numérico** frente a la verdad:

$$f(I) = \\Big[t_{ref} + \\tau_m\\ln\\frac{R_mI}{R_mI-(V_{th}-E_L)}\\Big]^{-1}$$"""),
    code(SETUP),
    code("print(inspect.getsource(ns.simulate_lif))"),
    code("""for I in (1.4, 2.0, 3.0):
    r = ns.simulate_lif(lambda t, I=I: I if t >= 10 else 0.0, 100, dt=0.05)
    plt.plot(r['t'], r['V'], label=f'{I} nA')
plt.axhline(ns.LIF_CELL['V_th'], color='#CC6677'); plt.legend(); plt.xlabel('time (ms)'); plt.ylabel('V (mV)'); plt.show()"""),
    code("""I = np.linspace(0, 5, 26)
sim = [ns.simulate_lif(i, 2000, dt=0.05)['spikes'].size / 2.0 for i in I]
plt.plot(I, ns.lif_rate_closed_form(I), label='exact'); plt.plot(I, sim, 's', mfc='white', label='simulation')
plt.xlabel('current (nA)'); plt.ylabel('rate (Hz)'); plt.legend(); plt.show()
print('rheobase =', (ns.LIF_CELL['V_th'] - ns.LIF_CELL['E_L']) / ns.LIF_CELL['R_m'], 'nA')"""),
    md("""## Exercises
1. Repeat the comparison with `dt=0.5` and `dt=1.0`. How large is the error? Why does a coarse step under-estimate the rate?
2. Derive the rheobase $I_{rh}=(V_{th}-E_L)/R_m$ from the steady state of the equation.
3. Set `params={'t_ref': 0}`. What limits the rate now?
4. Compare this f-I curve with Module 1. Name two behaviours of the Hodgkin-Huxley cell that LIF cannot show.""",
       """## Ejercicios
1. Repite la comparación con `dt=0.5` y `dt=1.0`. ¿Qué tan grande es el error? ¿Por qué un paso grueso subestima la frecuencia?
2. Deduce la reobase $I_{rh}=(V_{th}-E_L)/R_m$ a partir del estado estacionario de la ecuación.
3. Usa `params={'t_ref': 0}`. ¿Qué limita ahora la frecuencia?
4. Compara esta curva f-I con la del Módulo 1. Nombra dos comportamientos de la célula de Hodgkin-Huxley que el LIF no puede mostrar."""),
]

# --------------------------------------------------------------------- 03
NB["03_izhikevich"] = [
    md("""# Module 3 · Many firing patterns from four numbers (Izhikevich)

Neurons recorded in a slice are classified by how they fire: regular, in bursts, fast, adapting. Izhikevich (2003) showed that two equations reproduce most of these patterns:

$$\\frac{dv}{dt}=0.04v^2+5v+140-u+I,\\qquad \\frac{du}{dt}=a(bv-u),\\qquad v\\ge30 \\Rightarrow v\\leftarrow c,\\ u\\leftarrow u+d$$

$u$ is a slow recovery current; $a$ sets its speed, $b$ its coupling to $v$, $c$ the reset voltage and $d$ the after-spike kick.""",
       """# Módulo 3 · Muchos patrones de disparo con cuatro números (Izhikevich)

Las neuronas registradas en una rebanada se clasifican por cómo disparan: regular, en ráfagas, rápido, con adaptación. Izhikevich (2003) mostró que dos ecuaciones reproducen la mayoría de estos patrones:

$$\\frac{dv}{dt}=0.04v^2+5v+140-u+I,\\qquad \\frac{du}{dt}=a(bv-u),\\qquad v\\ge30 \\Rightarrow v\\leftarrow c,\\ u\\leftarrow u+d$$

$u$ es una corriente lenta de recuperación; $a$ fija su velocidad, $b$ su acoplamiento con $v$, $c$ el voltaje de reinicio y $d$ el impulso después de cada disparo."""),
    code(SETUP),
    code("print(inspect.getsource(ns.izhikevich_cell))"),
    code("""fig, ax = plt.subplots(5, 1, figsize=(7, 7), sharex=True)
for a_, (key, spec) in zip(ax, ns.IZH_CLASSES.items()):
    r = ns.izhikevich_cell(*spec['abcd'], lambda t: 10.0 * (t >= 20), duration=250)
    a_.plot(r['t'], r['v'], lw=0.8); a_.set_title(f"{key}: {spec['name']}", loc='left', fontsize=9)
ax[-1].set_xlabel('time (ms)'); plt.tight_layout(); plt.show()"""),
    md("""## Exercises
1. Move `c` from -65 to -50 in steps of 5 with the RS values. At which value does the cell start to burst?
2. Increase `d` in the FS cell. When does adaptation appear?
3. Run with `dt=1.0`. Which patterns survive?
4. Look up the *thalamo-cortical* and *resonator* parameter sets in Izhikevich (2003) and add them.""",
       """## Ejercicios
1. Mueve `c` de -65 a -50 en pasos de 5 con los valores de RS. ¿Con qué valor la célula empieza a disparar en ráfagas?
2. Aumenta `d` en la célula FS. ¿Cuándo aparece la adaptación?
3. Ejecuta con `dt=1.0`. ¿Qué patrones se mantienen?
4. Busca en Izhikevich (2003) los parámetros *tálamo-cortical* y *resonador* y agrégalos."""),
]

# --------------------------------------------------------------------- 04
NB["04_ei_network"] = [
    md("""# Module 4 · From cells to circuits: excitation, inhibition and synchrony

The cortex contains roughly 80 % excitatory and 20 % inhibitory neurons. Here 1,000 LIF cells are randomly connected (10 %) and driven by external noise. One parameter, the relative strength of inhibition $g$, decides whether the circuit fires in **synchronous volleys** (weak inhibition) or in an **asynchronous, irregular** way (strong inhibition), as the awake cortex does (Brunel 2000).

**Clinical hook.** Many seizures are thought to arise when inhibition fails and excitation recruits large populations at once. This module lets you watch that transition.

Measures: mean rate; CV of inter-spike intervals (0 = clock, 1 = random); synchrony index (variance/mean of the population count).""",
       """# Módulo 4 · De las células a los circuitos: excitación, inhibición y sincronía

La corteza tiene aproximadamente 80 % de neuronas excitatorias y 20 % de inhibitorias. Aquí 1000 células LIF se conectan al azar (10 %) y reciben ruido externo. Un parámetro, la fuerza relativa de la inhibición $g$, decide si el circuito dispara en **descargas síncronas** (inhibición débil) o de forma **asíncrona e irregular** (inhibición fuerte), como la corteza en vigilia (Brunel 2000).

**Gancho clínico.** Se piensa que muchas crisis epilépticas surgen cuando la inhibición falla y la excitación recluta poblaciones enteras a la vez. Este módulo permite observar esa transición.

Medidas: frecuencia media; CV de los intervalos entre disparos (0 = reloj, 1 = azar); índice de sincronía (varianza/media del conteo poblacional)."""),
    code(SETUP),
    code("""fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for a_, (g, ext) in zip(ax, [(2.0, 8.0), (6.0, 7.0)]):
    net = ns.simulate_ei_network(g=g, ext_rate=ext, duration=500)
    st = ns.network_statistics(net)
    m = net['times'] > 350
    a_.scatter(net['times'][m], net['ids'][m], s=0.4, marker='|',
               c=np.where(net['ids'][m] < net['n_exc'], '#44AA99', '#CC6677'))
    a_.set_title(f"g={g}: {st['rate_hz']:.0f} Hz, CV={st['cv_isi']:.2f}, sync={st['synchrony']:.0f}", fontsize=9)
    a_.set_xlabel('time (ms)')
ax[0].set_ylabel('neuron'); plt.show()"""),
    md("""## Exercises
1. Sweep `g` from 1 to 8 (three seeds each) and plot rate, CV and synchrony against `g`. Where is the transition?
2. With `g=6`, increase `ext_rate` to 12. Does irregular firing survive strong drive?
3. Simulate "loss of interneurons" by reducing `n_inh` to 100 while keeping `g=6`. What happens?
4. Why does irregular firing need inhibition? Explain in two sentences.""",
       """## Ejercicios
1. Recorre `g` de 1 a 8 (tres semillas por valor) y grafica frecuencia, CV y sincronía frente a `g`. ¿Dónde está la transición?
2. Con `g=6`, sube `ext_rate` a 12. ¿Se mantiene el disparo irregular con una entrada fuerte?
3. Simula una "pérdida de interneuronas" reduciendo `n_inh` a 100 con `g=6`. ¿Qué ocurre?
4. ¿Por qué el disparo irregular necesita inhibición? Explícalo en dos oraciones."""),
]

# --------------------------------------------------------------------- 05
NB["05_learning_machine"] = [
    md("""# Module 5 · A learning machine that recognises neuron types

An artificial neuron keeps three operations of a biological one: **weigh** its inputs, **add** them, and **decide** through a nonlinearity. It drops voltage, time and spikes, and gains a learning rule (backpropagation).

Here we close the loop with Module 3: we simulate 600 heterogeneous Izhikevich cells (five classes, random parameters, currents and noise), describe each spike train with five features, and train a small network to recognise the class, as an electrophysiologist would. A multinomial logistic regression (no hidden layer) is the statistical baseline.""",
       """# Módulo 5 · Una máquina que aprende a reconocer tipos de neurona

Una neurona artificial conserva tres operaciones de la biológica: **ponderar** sus entradas, **sumarlas** y **decidir** con una no linealidad. Pierde el voltaje, el tiempo y los disparos, y gana una regla de aprendizaje (retropropagación).

Aquí cerramos el círculo con el Módulo 3: simulamos 600 células de Izhikevich heterogéneas (cinco clases, parámetros, corrientes y ruido al azar), describimos cada tren de disparos con cinco rasgos y entrenamos una red pequeña para reconocer la clase, como lo haría un electrofisiólogo. Una regresión logística multinomial (sin capa oculta) sirve de referencia estadística."""),
    code(SETUP),
    code("""X, y, trains = ns.make_dataset(n_per_class=120, seed=0)
print(X.shape, 'features:', ns.FEATURES)
for k, key in enumerate(ns.CLASS_KEYS):
    print(f'{key:4s}', np.round(X[y == k].mean(0), 2))"""),
    code("print(inspect.getsource(ns.SpikeClassNet._layers))\nprint(inspect.getsource(ns.SpikeClassNet.fit))"),
    code("""train, test = ns.stratified_split(y, 0.3, seed=0)
net = ns.SpikeClassNet(5, 5, hidden=(16, 16), seed=0)
hist = net.fit(X[train], y[train], epochs=150, X_val=X[test], y_val=y[test])
base = ns.SpikeClassNet(5, 5, hidden=(), seed=0); base.fit(X[train], y[train], epochs=150)
for name, model in (('MLP', net), ('logistic', base)):
    pred = model.predict(X[test])
    print(f"{name:8s} accuracy {np.mean(pred == y[test]):.3f}  macro-F1 {ns.macro_f1(y[test], pred, 5):.3f}")
print(ns.confusion_matrix(y[test], net.predict(X[test]), 5))"""),
    code("""plt.semilogy(hist['train_loss'], label='training'); plt.semilogy(hist['val_loss'], '--', label='test')
plt.xlabel('epoch'); plt.ylabel('cross-entropy'); plt.legend(); plt.show()"""),
    md("""## An honest evaluation
One split is one draw. We repeat the split ten times; because the ten training sets share most of their cells, a naive interval is too narrow, so we use the corrected resampled t interval of Nadeau and Bengio. Then, because simulated data are free, we test both models on 600 **new** cells and compare them cell by cell (McNemar test) and by the quality of their probabilities (Brier score, calibration error).""",
       """## Una evaluación honesta
Una partición es un solo sorteo. Repetimos la partición diez veces; como los diez conjuntos de entrenamiento comparten la mayoría de sus células, un intervalo ingenuo es demasiado estrecho, así que usamos el intervalo t remuestreado corregido de Nadeau y Bengio. Después, como los datos simulados son gratuitos, probamos ambos modelos en 600 células **nuevas** y los comparamos célula a célula (prueba de McNemar) y por la calidad de sus probabilidades (puntuación de Brier, error de calibración)."""),
    code("""acc = {'MLP': [], 'LR': []}
for s in range(10):
    tr, te = ns.stratified_split(y, 0.3, seed=s)
    for name, hid in (('MLP', (16, 16)), ('LR', ())):
        m = ns.SpikeClassNet(5, 5, hid, seed=s); m.fit(X[tr], y[tr], epochs=150, seed=s)
        acc[name].append(np.mean(m.predict(X[te]) == y[te]))
for name, v in acc.items():
    c = ns.corrected_resampled_t(v, 420, 180)
    naive = 1.96 * np.std(v, ddof=1) / np.sqrt(10)
    print(f"{name}: {c['mean']:.3f}  corrected 95% CI {c['lower']:.3f}-{c['upper']:.3f}  (naive +/-{naive:.3f})")
d = ns.corrected_resampled_t(np.subtract(acc['MLP'], acc['LR']), 420, 180)
print(f"difference {d['mean']:+.3f}, 95% CI {d['lower']:+.3f} to {d['upper']:+.3f}, p = {d['p']:.2f}")"""),
    code("""Xn, yn, _ = ns.make_dataset(n_per_class=120, seed=1)          # 600 new cells
mlp = ns.SpikeClassNet(5, 5, (16, 16), seed=0); mlp.fit(X, y, epochs=150, seed=0)
lr  = ns.SpikeClassNet(5, 5, (), seed=0);       lr.fit(X, y, epochs=150, seed=0)
for name, m in (('MLP', mlp), ('LR', lr)):
    P = m.predict_proba(Xn); ece = ns.calibration_curve(P, yn)[3]
    print(f"{name}: accuracy {np.mean(P.argmax(1) == yn):.3f}  Brier {ns.brier_score(P, yn):.3f}  ECE {ece:.3f}")
print('McNemar:', ns.mcnemar_exact(mlp.predict(Xn) == yn, lr.predict(Xn) == yn))"""),
    md("""## Exercises
1. Compare the corrected and the naive intervals above. Why is the naive one about three times narrower, and which would you trust? Is the MLP really better than logistic regression?
2. Train with only two features (burst fraction and CV). Which classes become confused, and why?
3. Remove the ReLU (make the hidden layers linear). What does the network become?
4. The test loss creeps up after a few dozen epochs while test accuracy stays flat. What is this called, and how would you prevent it?
5. List what the artificial neuron keeps and discards compared with Modules 1-3.""",
       """## Ejercicios
1. Compara los intervalos corregido e ingenuo de arriba. ¿Por qué el ingenuo es unas tres veces más estrecho y en cuál confiarías? ¿La red es realmente mejor que la regresión logística?
2. Entrena solo con dos rasgos (fracción en ráfaga y CV). ¿Qué clases se confunden y por qué?
3. Quita la ReLU (capas ocultas lineales). ¿En qué se convierte la red?
4. La pérdida de prueba sube poco a poco después de unas decenas de épocas mientras la exactitud de prueba se mantiene. ¿Cómo se llama esto y cómo lo evitarías?
5. Haz una lista de lo que la neurona artificial conserva y descarta frente a los Módulos 1 a 3."""),
]

# --------------------------------------------------------------------- 06
NB["06_clinical_scenarios"] = [
    md("""# Module 6 · Clinical "what-if" experiments

These experiments use the Module 1 model to illustrate **mechanisms**, not to predict patient values (the parameters come from the squid axon).

| Scenario | Model change | Clinical counterpart |
|---|---|---|
| Na⁺ channel block | lower `gNa` | lidocaine, phenytoin, carbamazepine; TTX poisoning |
| K⁺ channel block | lower `gK` | 4-aminopyridine (dalfampridine) in multiple sclerosis |
| Hyperkalaemia | shift `EK` (Nernst) | renal failure, rhabdomyolysis; depolarisation block |
| Loss of inhibition | lower `g` in Module 4 | seizures, GABAergic dysfunction |""",
       """# Módulo 6 · Experimentos clínicos de tipo "¿qué pasaría si...?"

Estos experimentos usan el modelo del Módulo 1 para ilustrar **mecanismos**, no para predecir valores de pacientes (los parámetros vienen del axón de calamar).

| Escenario | Cambio en el modelo | Equivalente clínico |
|---|---|---|
| Bloqueo de canales de Na⁺ | bajar `gNa` | lidocaína, fenitoína, carbamazepina; intoxicación por TTX |
| Bloqueo de canales de K⁺ | bajar `gK` | 4-aminopiridina (dalfampridina) en esclerosis múltiple |
| Hiperpotasemia | desplazar `EK` (Nernst) | insuficiencia renal, rabdomiólisis; bloqueo por despolarización |
| Pérdida de inhibición | bajar `g` en el Módulo 4 | crisis epilépticas, disfunción GABAérgica |"""),
    code(SETUP),
    code("""I = np.arange(0, 30.1, 1.0)
for block in (0.0, 0.1, 0.2, 0.3):
    plt.plot(I, ns.fi_curve(I, ns.sodium_block(block), duration=400), 'o-', ms=2.5, label=f'{block:.0%} Na⁺ block')
plt.xlabel('current (µA/cm²)'); plt.ylabel('rate (Hz)'); plt.legend(); plt.show()"""),
    code("""for block in (0.0, 0.2, 0.3):
    w, r = ns.spike_width(ns.potassium_block(block))
    plt.plot(r['t'], r['V'], label=f'{block:.0%} K⁺ block: width {w:.2f} ms')
plt.xlim(4, 16); plt.xlabel('time (ms)'); plt.ylabel('V (mV)'); plt.legend(); plt.show()"""),
    code("""for K in (4, 7, 12):
    p = ns.extracellular_potassium(K)
    r = ns.simulate_hh(0.0, 150, params=p, init=ns.settle(p))
    plt.plot(r['t'], r['V'], label=f'[K⁺]o = {K} mM (E_K = {p["EK"]:.0f} mV)')
plt.xlabel('time (ms)'); plt.ylabel('V (mV)'); plt.legend(); plt.show()"""),
    md("""## Exercises
1. **Use-dependence.** Real lidocaine blocks more during fast firing. Make `gNa` decrease by 2 % after each spike and recover slowly. Does high-frequency firing fail first?
2. **Safety margin.** What fraction of Na⁺ block abolishes repetitive firing at 20 µA/cm²?
3. **4-AP and seizures.** Increase the K⁺ block to 40-50 %. Why might a drug that improves conduction also lower the seizure threshold?
4. **Hyperkalaemia.** Find the [K⁺]o at which spontaneous firing appears and the value at which it stops. Explain the second transition with the gate `h`.
5. **Hypokalaemia.** Try [K⁺]o = 2.5 mM. What happens to excitability?""",
       """## Ejercicios
1. **Dependencia del uso.** La lidocaína real bloquea más durante el disparo rápido. Haz que `gNa` baje 2 % después de cada disparo y se recupere lentamente. ¿Falla primero el disparo de alta frecuencia?
2. **Margen de seguridad.** ¿Qué fracción de bloqueo de Na⁺ suprime el disparo repetitivo con 20 µA/cm²?
3. **4-AP y crisis.** Aumenta el bloqueo de K⁺ a 40-50 %. ¿Por qué un fármaco que mejora la conducción podría también bajar el umbral convulsivo?
4. **Hiperpotasemia.** Encuentra la [K⁺]o con la que aparece el disparo espontáneo y el valor con el que se detiene. Explica la segunda transición con la compuerta `h`.
5. **Hipopotasemia.** Prueba [K⁺]o = 2,5 mM. ¿Qué le pasa a la excitabilidad?"""),
]

# --------------------------------------------------------------------- 07
NB["07_choosing_a_model"] = [
    md("""# Module 7 · Choosing a model: realism versus cost

No model is "best". The right level of detail depends on the question. This module measures, on your own computer, how much each single-cell model costs.""",
       """# Módulo 7 · Cómo elegir un modelo: realismo frente a costo

Ningún modelo es "el mejor". El nivel de detalle adecuado depende de la pregunta. Este módulo mide, en tu propia computadora, cuánto cuesta cada modelo de neurona."""),
    code(SETUP),
    code("""import time
def best_time(f, reps=3):
    out = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); f(); out = min(out, time.perf_counter() - t0)
    return out

runs = {'Hodgkin-Huxley (dt 0.01 ms)': lambda: ns.simulate_hh(10.0, 1000, 0.01),
        'LIF (dt 0.1 ms)': lambda: ns.simulate_lif(2.0, 1000, 0.1),
        'Izhikevich (dt 0.25 ms)': lambda: ns.izhikevich_cell(*ns.IZH_CLASSES['RS']['abcd'], 10.0, 1000, 0.25)}
for name, f in runs.items():
    print(f'{name:30s} {best_time(f):.4f} s per simulated second')"""),
    md("""## Exercises
1. Run Hodgkin-Huxley with `dt=0.05` and `dt=0.1`. Compare spike times with `dt=0.01`. When does it break?
2. Estimate how long 10,000 HH cells would take for 1 s of biological time with this code, and with a vectorised version.
3. Write one paragraph justifying a model for each question: (a) a new Na⁺-channel antiseizure drug; (b) working memory in a network of 10⁵ cells; (c) automatic classification of interictal EEG spikes.""",
       """## Ejercicios
1. Ejecuta Hodgkin-Huxley con `dt=0.05` y `dt=0.1`. Compara los tiempos de disparo con `dt=0.01`. ¿Cuándo falla?
2. Estima cuánto tardarían 10 000 células HH en simular 1 s de tiempo biológico con este código y con una versión vectorizada.
3. Escribe un párrafo que justifique un modelo para cada pregunta: (a) un nuevo antiepiléptico que bloquea canales de Na⁺; (b) memoria de trabajo en una red de 10⁵ células; (c) la clasificación automática de puntas interictales en el EEG."""),
]


def build(lang, outdir):
    out = Path(outdir); out.mkdir(exist_ok=True)
    for old in out.glob("*.ipynb"):
        old.unlink()
    for name, cells in NB.items():
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        for kind, content in cells:
            nb.cells.append(nbf.v4.new_markdown_cell(content[lang]) if kind == "md"
                            else nbf.v4.new_code_cell(content))
        nbf.write(nb, out / f"{name}.ipynb")


if __name__ == "__main__":
    build("en", "notebooks")
    build("es", "notebooks_es")
    print("notebooks built")
