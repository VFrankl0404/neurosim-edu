"""Number citations by first appearance and write the reference list.
Input: manuscript_keyed.md with [@key;@key] citations; references_vancouver.md
(literature review) plus EXTRA below. Output: ../manuscript_en.md"""
import re, sys
from pathlib import Path
here = Path(__file__).parent
src = (here / "manuscript_keyed.md").read_text()
refs = {}
for block in re.split(r"\n(?=\d+\. )", (here / "references_vancouver.md").read_text()):
    m = re.match(r"\d+\. (.*?)\n.*key `([^`]+)`", block, re.S)
    if m:
        refs[m.group(2)] = m.group(1).strip()
EXTRA = {
 "nadeau03": "Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn. 2003;52(3):239-81. doi:10.1023/A:1024068626366.",
 "mcnemar47": "McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika. 1947;12(2):153-7. doi:10.1007/BF02295996. PMID: 20254758.",
 "brier50": "Brier GW. Verification of forecasts expressed in terms of probability. Mon Weather Rev. 1950;78(1):1-3. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.",
 "guo17": "Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning. PMLR. 2017;70:1321-30.",
 # v2.1 additions (verified against PubMed records, 2026-11-20; brooke96 is a book chapter not indexed in PubMed)
 "dragly17": "Dragly SA, Hobbi Mobarhan M, Våvang Solbrå A, Tennøe S, Hafreager A, Malthe-Sørenssen A, et al. Neuronify: an educational simulator for neural circuits. eNeuro. 2017;4(2):ENEURO.0022-17.2017. doi:10.1523/ENEURO.0022-17.2017. PMID: 28321440. PMCID: PMC5355897.",
 "spreizer21": "Spreizer S, Senk J, Rotter S, Diesmann M, Weyers B. NEST Desktop, an educational application for neuroscience. eNeuro. 2021;8(6):ENEURO.0274-21.2021. doi:10.1523/ENEURO.0274-21.2021. PMID: 34764188. PMCID: PMC8638679.",
 "gorur18": "Gorur-Shandilya S, Hoyland A, Marder E. Xolotl: an intuitive and approachable neuron and network simulator for research and teaching. Front Neuroinform. 2018;12:87. doi:10.3389/fninf.2018.00087. PMID: 30534067. PMCID: PMC6275287.",
 "rajbanshi22": "Rajbanshi B, Guruacharya A. Panama: an open-source educational app for ion channel biophysics simulation. Front Neuroinform. 2022;16:813940. doi:10.3389/fninf.2022.813940. PMID: 35356446. PMCID: PMC8959878.",
 "heitler22": "Heitler B. Neurosim: some thoughts on using computer simulation in teaching electrophysiology. J Undergrad Neurosci Educ. 2022;20(2):A284-91. doi:10.59390/JGIP5297. PMID: 38323048. PMCID: PMC10653224.",
 "monteiro21": "Monteiro O, Bhaskar A, Wong IN, Ng AKM, Baptista-Hon DT. Teaching bioelectricity and neurophysiology to medical students using LabAXON simulations. Adv Physiol Educ. 2021;45(4):702-8. doi:10.1152/advan.00054.2021. PMID: 34498935.",
 "hopper22": "Hopper AJ, Beswick-Jones H, Brown AM. A color-coded graphical guide to the Hodgkin and Huxley papers. Adv Physiol Educ. 2022;46(4):580-92. doi:10.1152/advan.00178.2022. PMID: 36007940.",
 "hodgkin49": "Hodgkin AL, Katz B. The effect of temperature on the electrical activity of the giant axon of the squid. J Physiol. 1949;109(1-2):240-9. doi:10.1113/jphysiol.1949.sp004388. PMID: 15394322. PMCID: PMC1392577.",
 "frankenhaeuser57": "Frankenhaeuser B, Hodgkin AL. The action of calcium on the electrical properties of squid axons. J Physiol. 1957;137(2):218-44. doi:10.1113/jphysiol.1957.sp005808. PMID: 13449874. PMCID: PMC1362975.",
 "brooke96": "Brooke J. SUS: a 'quick and dirty' usability scale. In: Jordan PW, Thomas B, Weerdmeester BA, McClelland IL, editors. Usability evaluation in industry. London: Taylor & Francis; 1996. p. 189-94.",
 "irnich02":"Irnich W. Georges Weiss' fundamental law of electrostimulation is 100 years old. Pacing Clin Electrophysiol. 2002;25(2):245-8. doi:10.1046/j.1460-9592.2002.00245.x. PMID: 11916000.",
}
refs.update(EXTRA)
order = []
def num(keys):
    for k in keys:
        if k not in refs: sys.exit(f"missing reference: {k}")
        if k not in order: order.append(k)
    ns = sorted(order.index(k) + 1 for k in keys)
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        out.append(str(ns[i]) if j == i else (f"{ns[i]},{ns[j]}" if j == i + 1 else f"{ns[i]}–{ns[j]}"))
        i = j + 1
    return "[" + ",".join(out) + "]"
body = re.sub(r"\[(@[^\]]+)\]", lambda m: num([k.strip().lstrip("@") for k in m.group(1).split(";")]), src)
reflist = "\n\n".join(f"{i}. {refs[k]}" for i, k in enumerate(order, 1))
body = body.replace("[[REFERENCES]]", reflist)
(here.parent / "manuscript_en.md").write_text(body)
unused = sorted(set(refs) - set(order))
print("cited:", len(order), "| unused:", unused)
(here / "citation_order.txt").write_text("\n".join(f"{i} {k}" for i, k in enumerate(order, 1)))
