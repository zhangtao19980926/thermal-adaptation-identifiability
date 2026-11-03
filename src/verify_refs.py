# -*- coding: utf-8 -*-
"""Verify candidate references against Crossref (DOI / bibliographic).
Run with thermal env (urllib only). Writes outputs/refs_verified.csv and .txt.
"""
import os, json, time, urllib.parse, urllib.request, difflib, io
import pandas as pd

ROOT = r"D:\thermal-adaptation-series\p1\outputs"
MAILTO = "liruiqi1@gxcme.edu.cn"

# (id, doi_hint or "", title query, note)
REFS = [
 # --- adaptive theory / classics ---
 ("humphreys1978","10.1080/09613217808550656","Outdoor temperatures and comfort indoors",""),
 ("auliciems1981","10.1007/BF02184458","Towards a psycho-physiological model of thermal perception",""),
 ("brager_dedear1998","10.1016/S0378-7788(97)00053-4","Thermal adaptation in the built environment a literature review",""),
 ("dedear_brager2002","10.1016/S0378-7788(02)00005-1","Thermal comfort in naturally ventilated buildings revisions to ASHRAE Standard 55",""),
 ("nicol_humphreys2002","10.1016/S0378-7788(02)00006-3","Adaptive thermal comfort and sustainable thermal standards for buildings",""),
 ("cabanac1971","10.1126/science.173.4002.1103","Physiological role of pleasure",""),
 ("yao2009apmv","10.1016/j.buildenv.2009.02.014","A theoretical adaptive model of thermal comfort Adaptive Predicted Mean Vote aPMV",""),
 ("arens2010","10.1016/j.buildenv.2009.03.002","Are class A temperature requirements realistic or desirable",""),
 ("halawa_vanhoof2012","10.1016/j.enbuild.2012.04.011","The adaptive approach to thermal comfort A critical overview",""),
 ("nguyen2012","10.1016/j.buildenv.2012.03.021","An adaptive thermal comfort model for hot humid South-East Asia",""),
 ("toe_kubota2013","","Development of an adaptive thermal comfort equation for naturally ventilated buildings in hot humid climates using ASHRAE RP-884 database","Frontiers Arch Research 2013"),
 ("dedear2013progress","10.1111/ina.12046","Progress in thermal comfort research over the last twenty years","Indoor Air"),
 ("rupp2015","10.1016/j.enbuild.2015.07.047","A review of human thermal comfort in the built environment",""),
 ("parkinson_dedear2015","","Thermal pleasure in built environments physiology of alliesthesia","Build Res Inf 2015"),
 ("luo2015mixed","","Evaluating thermal comfort in mixed-mode buildings A field study in a subtropical climate","Build Environ 2015"),
 ("enescu2017","10.1016/j.rser.2017.05.175","A review of thermal comfort models and indicators for indoor environments",""),
 ("vanhoof2017ageing","10.1016/j.buildenv.2017.05.008","Ten questions concerning thermal comfort and ageing",""),
 ("carlucci2018","","Review of adaptive thermal comfort models in built environmental regulatory documents","Build Environ 2018"),
 ("wangz2018inddiff","","Individual difference in thermal comfort A literature review","Build Environ 2018"),
 ("schellen2010","10.1111/j.1600-0668.2010.00657.x","Differences between young adults and elderly in thermal comfort productivity and thermal physiology in response to a moderate temperature drift and a steady-state condition",""),
 ("dedear2020review","10.1016/j.enbuild.2020.109893","A review of adaptive thermal comfort research since 1998",""),
 ("schweiker2020multidomain","10.1016/j.buildenv.2020.106804","Review of multi-domain approaches to indoor environmental perception and behaviour",""),
 ("schweiker2015athb","","A framework for an adaptive thermal heat balance model ATHB","Build Environ 2015"),
 # --- databases / PCM / tools ---
 ("foldvary2018db2","10.1016/j.buildenv.2018.06.022","Development of the ASHRAE Global Thermal Comfort Database II",""),
 ("cheung2019pmv","10.1016/j.buildenv.2019.01.055","Analysis of the accuracy on PMV PPD model using the ASHRAE Global Thermal Comfort Database II",""),
 ("yang2023ctcd","10.1038/s41597-023-02568-3","The Chinese thermal comfort dataset","Sci Data 10 662"),
 ("tartarini2022dorn","10.1111/ina.13160","Personal comfort models based on a 6-month experiment using environmental parameters and data from wearables","Indoor Air"),
 ("gao2022ingauge","","Understanding occupants behaviour engagement emotion and comfort indoors with heterogeneous sensors and wearables","Sci Data"),
 ("lius2019wearable","10.1016/j.buildenv.2019.106281","Personal thermal comfort models with wearable sensors",""),
 ("kim2018pcm_new","10.1016/j.buildenv.2018.01.023","Personal comfort models A new paradigm in thermal comfort for occupant-centric environmental control",""),
 ("kim2018pcm_behavior","10.1016/j.buildenv.2017.12.011","Personal comfort models Predicting individuals thermal preference using occupant heating and cooling behavior and machine learning",""),
 ("tartarini2020cbe","10.1016/j.softx.2020.100563","CBE Thermal Comfort Tool online tool for thermal comfort calculations and visualizations",""),
 ("cozie2019jpcs","10.1088/1742-6596/1343/1/012145","A smartwatch methodology for the in-situ collection of occupant comfort data","J Phys Conf Ser"),
 ("dedear2015schools","10.1080/09613218.2015.991627","Adaptive thermal comfort in Australian school classrooms",""),
 # --- occupant behaviour / annex ---
 ("obrien2020annex79","10.1016/j.buildenv.2020.106738","Introducing IEA EBC Annex 79 Key challenges and opportunities in the field of occupant-centric building design and operation",""),
 ("gunay2013","","A state-of-the-art review of occupant behavior and energy efficiency in buildings","Build Environ 2013"),
 ("hong2017tenq","10.1016/j.buildenv.2016.12.006","Ten questions concerning occupant behavior in buildings The big picture",""),
 ("yan2015","10.1016/j.enbuild.2015.08.032","Occupant behavior modeling for building performance simulation Current state and future challenges",""),
 ("haldi2008","","On the behaviour and adaptations of office occupants","Build Environ 2008"),
 ("daum2011","10.1016/j.buildenv.2010.06.012","A personalized measure of thermal comfort for building controls",""),
 # --- physiology / transient models ---
 ("tanabe2002_65mn","10.1016/S0378-7788(02)00014-2","Evaluation of thermal comfort using combined multi-node thermoregulation 65MN and radiation models and computational fluid dynamics",""),
 ("huizenga2001ucberkeley","10.1016/S0360-1323(00)00061-5","A model of human physiology and comfort for assessing complex thermal environments",""),
 ("zhangh2010_p1","10.1016/j.buildenv.2009.06.018","Thermal sensation and comfort models for non-uniform and transient environments part I local sensation of individual body parts",""),
 ("zhangh2010_p2","10.1016/j.buildenv.2009.06.019","Thermal sensation and comfort models for non-uniform and transient environments part II local comfort of individual body parts",""),
 ("zhangh2010_p3","10.1016/j.buildenv.2009.06.020","Thermal sensation and comfort models for non-uniform and transient environments part III whole-body sensation and comfort",""),
 ("kingma2015female","10.1038/nclimate2743","Energy consumption in buildings and female thermal demand",""),
 ("wangz2022oldersteps","","Chinese older peoples subjective and physiological responses to moderate cold and warm temperature steps","Build Environ"),
 ("kenney2017adolph","","Edward F Adolph Distinguished Lecture Skin-deep insights into vascular aging","J Appl Physiol"),
 ("holowatz2010skinblood","","Aging and the control of human skin blood flow","Front Biosci"),
 ("greaney2021vasodilation","10.1080/23328940.2020.1832950","Thermoregulatory reflex control of cutaneous vasodilation in healthy aging","Temperature"),
 ("daanen2018acclim","10.1007/s40279-017-0808-x","Heat Acclimation Decay and Re-Induction A Systematic Review and Meta-Analysis","Sports Med"),
 ("periard2016cv","","Cardiovascular adaptations supporting human exercise-heat acclimation an autonomous update","Exp Physiol"),
 ("sawka2011comprphysiol","","Integrated physiological mechanisms of exercise performance adaptation and maladaptation to heat stress","Compr Physiol"),
 ("coull2021agesensitivity","","Age comparison of changes in local warm and cold sensitivity due to whole body cooling","J Therm Biol"),
 ("martin2023trpv3","","Aging is associated with impaired triggering of TRPV3-mediated cutaneous vasodilation a crucial process for local heat exposure","GeroScience"),
 # --- running mean / setpoint / climate ---
 ("bienvenido2020alpha","","Optimization of energy saving with adaptive setpoint temperatures by calculating the prevailing mean outdoor air temperature","Build Environ 2020"),
 ("bienvenido2021expect","","Influence of the improvement in thermal expectation levels with adaptive setpoint temperatures on energy consumption","Appl Sci"),
 ("liuj2012behaviour","","Occupants behavioural adaptation in workplaces with non-central heating and cooling systems","Appl Therm Eng 2012"),
 ("vellei2017humidity","","The influence of relative humidity on adaptive thermal comfort","Build Environ 2017"),
 # --- methods: identifiability / AI / cohort ---
 ("raue2009ident","10.1093/bioinformatics/btp358","Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood",""),
 ("ngarambe2020ai","10.1016/j.enbuild.2020.109806","The use of artificial intelligence AI methods in the prediction of thermal comfort in buildings energy implications of AI-based thermal comfort controls",""),
 ("quintana2023cohort","","Cohort comfort models Using occupants similarity to predict personal thermal preference with less data","Build Environ"),
]

MANUAL = [
 ("fanger1970","Fanger, P.O. (1970). Thermal Comfort: Analysis and Applications in Environmental Engineering. Copenhagen: Danish Technical Press.","book, no DOI"),
 ("dedear_brager1998rp884","de Dear, R.J., Brager, G.S. (1998). Developing an adaptive model of thermal comfort and preference. ASHRAE Transactions, 104(1), 145-167. (RP-884)","ASHRAE Trans, no DOI"),
 ("gagge1971","Gagge, A.P., Stolwijk, J.A.J., Nishi, Y. (1971). An effective temperature scale based on a simple model of human physiological regulatory response. ASHRAE Transactions, 77(1), 247-262.","ASHRAE Trans"),
 ("gagge1986","Gagge, A.P., Fobelets, A.P., Berglund, L.G. (1986). A standard predictive index of human response to the thermal environment. ASHRAE Transactions, 92(2B), 709-731.","ASHRAE Trans"),
 ("ashrae55_2020","ANSI/ASHRAE Standard 55-2020 (2020). Thermal Environmental Conditions for Human Occupancy. Atlanta, GA: ASHRAE.","standard"),
 ("en16798_2019","CEN (2019). EN 16798-1:2019, Energy performance of buildings - Ventilation for buildings - Part 1: Indoor environmental input parameters for design and assessment of energy performance of buildings. Brussels: CEN.","standard"),
 ("iso7730_2005","ISO (2005). ISO 7730:2005, Ergonomics of the thermal environment - Analytical determination and interpretation of thermal comfort using PMV/PPD and local thermal comfort. Geneva: ISO.","standard"),
 ("chen2018node","Chen, R.T.Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. (2018). Neural Ordinary Differential Equations. Advances in Neural Information Processing Systems 31 (NeurIPS 2018). arXiv:1806.07366.","NeurIPS/arXiv"),
 ("rubanova2019latent","Rubanova, Y., Chen, R.T.Q., Duvenaud, D. (2019). Latent ODEs for Irregularly-Sampled Time Series. NeurIPS 2019 Workshop; arXiv:1907.03907.","arXiv"),
 ("kidger2020ncde","Kidger, P., Morrill, J., Foster, J., Lyons, T. (2020). Neural Controlled Differential Equations for Irregular Time Series. Advances in Neural Information Processing Systems 33 (NeurIPS 2020). arXiv:2005.08926.","NeurIPS/arXiv"),
]

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent":"ref-verify/1.0 (mailto:%s)"%MAILTO,"Accept":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def authors(m):
    au = m.get("author", [])
    fam = [a.get("family","") for a in au[:3]]
    return "; ".join(fam) + (" et al." if len(au)>3 else "")

rows = []
def add(rid,status,title,year,container,doi,au):
    rows.append(dict(id=rid,status=status,returned_title=title,year=year,container=container,doi=doi,authors=au))

for rid, doi_hint, qtitle, note in REFS:
    try:
        if doi_hint:
            d = get("https://api.crossref.org/works/%s" % urllib.parse.quote(doi_hint))
            m = d["message"]
            rt = (m.get("title") or [""])[0]
            sim = difflib.SequenceMatcher(None, qtitle.lower()[:40], rt.lower()[:40]).ratio()
            status = "ok_doi" if sim>0.5 else "CHECK_doi_title"
            add(rid,status,rt,m.get("published-print",m.get("published-online",{})).get("date-parts",[[""]])[0][0],
                (m.get("container-title") or [""])[0], m.get("DOI",""), authors(m))
        else:
            url = "https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(qtitle)
            d = get(url)
            best=None;bs=-1
            for it in d["message"]["items"]:
                rt=(it.get("title") or [""])[0]
                s=difflib.SequenceMatcher(None, qtitle.lower(), rt.lower()).ratio()
                if s>bs: bs,best=s,it
            rt=(best.get("title") or [""])[0]
            status = "ok_match" if bs>0.72 else ("weak_%.2f"%bs)
            add(rid,status,rt,best.get("published-print",best.get("published-online",{})).get("date-parts",[[""]])[0][0],
                (best.get("container-title") or [""])[0], best.get("DOI",""), authors(best))
    except Exception as e:
        add(rid,"ERROR:%s"%str(e)[:40],"","","","","")
    time.sleep(0.3)

for rid, text, note in MANUAL:
    add(rid,"manual",text,"","","","")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT,"refs_verified.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(ROOT,"refs_verified.txt"),"w",encoding="utf-8") as f:
    for _,r in df.iterrows():
        f.write("[%s] %s | %s | %s | %s | DOI:%s | %s\n" % (r.id,r.status,r.year,r.container,r.returned_title[:95],r.doi,r.authors))
print(df[["id","status","year","doi"]].to_string(index=False))
print("\nWEAK/CHECK/ERROR rows:")
print(df[~df.status.str.startswith(("ok","manual"))][["id","status","returned_title","doi"]].to_string(index=False))
print("\nWROTE refs_verified.csv / .txt ; total =", len(df))
