# -*- coding: utf-8 -*-
from docx import Document

def text(p):
    return "\n".join(x.text for x in Document(p).paragraphs)

m = text(r"D:\thermal-adaptation-series\p1\outputs\P1_manuscript_BE.docx")
c = text(r"D:\thermal-adaptation-series\p1\outputs\P1_cover_letter_BE.docx")

fails = []
def chk(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        fails.append(label)

for s in [
    # confident framing retained
    "that scope is itself a result",
    "divide the inferential labour",
    "boundaries are operational rather than restrictive",
    "the binding constraint is information rather than model capacity",
    "makes the identifiability boundary explicit",
    "external value and climate dependence of outdoor memory",
    "too little to support the whole-building energy savings",
    "methodological as well as complementary",
    "an identifiability result rather than a physiological one",
    "scoped to set-point timing rather than whole-building energy",
    "are not interpreted as effects",
    "we report alongside the MSE result",
    "the only correlate reaching significance",
    "information, not biology",
    "4.6 Scope and identifiability boundaries",
    # strengthened (evidence-backed) phrasing
    "for every time constant we estimate, we report both its value and the conditions under which it is recoverable",
    "formal identifiability profiling for the slow constant and uncertainty-and-robustness analysis for the fast one",
    "three converging lines of evidence agree across very different designs",
    "recovers the constant directly in real data (20-23 d)",
    "identify the optimal window within the dataset's native discrete scales",
    "about 15 days overall and 30 days for air-conditioned and cold-zone buildings",
    "identifies age as the one significant correlate of fast adaptation (mechanistically anchored in Section 4.4)",
    # data-faithful / hard qualifications that must stay
    "statistically indistinguishable from zero (p = 0.52, R2 = 0.06, ten cities)",
    "these city-level slopes are descriptive",
    "small and statistically indistinguishable from zero after quality control",
    "rather than arbitrary or causal window estimation",
    # hard facts
    "0.86% of occupied hours",
    "a median of about one day",
    "core-temperature or sweat measurements",
    "70,000 records",
    "-50.7",
]:
    chk("MS contains: " + s, s in m)

for s in [
    "Several limits bound the claims",
    "partly cautionary",
    "rather than conceal",
    "makes the failure mode explicit",
    "fails on the impoverished one",
    "flag rather than hide",
    "a limitation the longitudinal analyses",
    "do not claim whole-building energy",
    "is unstable and is discarded",
    "null or suggestive",
    "4.6 Limitations",
    # over-strong / data-mismatched / superseded phrasing that must be gone
    "of every time constant we estimate",
    "not estimable",
    "established by the convergence of three independent",
    "resolve the window",
    "sufficient to establish the age association",
    "specify exactly",
    "discarded as unstable",
    "three complementary lines of differing evidentiary weight",
    "identify an empirical optimum among the dataset's native",
    "supports an age association but is underpowered",
    "consistent with the known slowing of autonomic response with age",
]:
    chk("MS removed: " + s, s not in m)

n_exp = m.count("treated as exploratory")
chk("MS 'treated as exploratory' count == 1 (methods only), got %d" % n_exp, n_exp == 1)

for s in ["established practice in the field", "These are independent works",
          "can and cannot be recovered"]:
    chk("CL contains: " + s, s in c)
for s in ["mancripts", "Please insert here", "rather than conceals",
          "TO BE CONFIRMED", "partly cautionary"]:
    chk("CL removed: " + s, s not in c)

print("RESULT:", "ALL PASS" if not fails else ("%d FAIL -> %s" % (len(fails), fails)))
