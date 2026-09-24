import json, sys, importlib.util
import numpy as np
import pathlib
D=str(pathlib.Path(__file__).resolve().parents[2] / "wcont_control_read_2026-09-20/scripts") + "/"
def load(n):
    s=importlib.util.spec_from_file_location(n,D+n+".py"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
U=load("wcont_untaught_delta"); A=load("wcont_away_anchor")
d=json.load(open(pathlib.Path(__file__).resolve().parents[1] / "out/untaught/untaught_registry.json"))
teams,per,fin,win,res=U.per_team(d)
print("armW",win["armW"].sum(),fin["armW"].sum(),"wcont_p12M",win["wcont_p12M"].sum())
for r in ("wcontb_p3M","wcontb_p6M","wcontb_p12M"):
    pt,lo,hi=U.paired(per[r],per["armW"]); v=U.verdict(pt*100 if abs(pt)<1.5 else pt,(lo*100 if abs(pt)<1.5 else lo,hi*100 if abs(pt)<1.5 else hi),3.69,"ahead","behind")
    print(r,v["delta_pp"],v["ci95_pp"],v["verdict"],"teams+",int((per[r]>per["armW"]).sum()))
for a,b in (("wcontb_p12M","wcont_p12M"),("wcontb_p12M","wcontb_p3M")):
    pt,lo,hi=U.paired(per[a],per[b]); s=100 if abs(pt)<1.5 else 1
    print(a,"-",b,round(pt*s,2),[round(lo*s,2),round(hi*s,2)],"teams+",int((per[a]>per[b]).sum()))
for name,k in (("armW",50),("wcont",64)):
    dd,lo,hi=A.newcombe(71,100,k,100); print("anchor wcontb -",name,round(dd,3),[round(lo,3),round(hi,3)])
print("wilson",A.wilson(71,100))
