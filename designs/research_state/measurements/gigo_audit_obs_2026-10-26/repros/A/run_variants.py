"""Run the hidden-fact variants (investigator A). p1 = Tyranitar-lead pool team; p2 base = a Skarmory lead whose
policy is 'move slot 1 (Spikes) when legal'. Each variant changes ONE hidden fact of p2's team."""
from __future__ import annotations

import sys

import numpy as np

import leak_diff as L

P1 = ("Tyranitar||leftovers|sandstream|crunch,earthquake,icebeam,pursuit|Modest|252,,,252,4,|||||]"
      "Skarmory||leftovers|keeneye|spikes,roar,toxic,protect|Careful|252,,4,,252,||,0,,,,|||]"
      "Moltres||leftovers|pressure|willowisp,flamethrower,hiddenpowergrass,protect|Timid|,,,252,4,252||,2,,30,,|||]"
      "Jirachi||leftovers|serenegrace|toxic,firepunch,wish,protect|Calm|248,,92,,132,36||,0,,,,|||]"
      "Gengar||leftovers|levitate|destinybond,taunt,willowisp,thunderbolt|Hasty|168,,,,164,176||,0,,,,|||]"
      "Flygon||leftovers|levitate|earthquake,toxic,protect,rockslide|Impish|252,,224,,,32|||||")

SK = "Skarmory||{item}|{ability}|spikes,protect,roar,{m4}|{nature}|252,,8,{spa},248,|F|{ivs}|||"
REST = ["Blissey||leftovers|naturalcure|softboiled,icebeam,toxic,fireblast|Modest|,,252,252,,4||,0,,,,|S||",
        "Tyranitar||leftovers|sandstream|focuspunch,rockslide,hiddenpowerbug,earthquake|Adamant|248,196,12,,52,|F|,30,30,,30,|||",
        "Swampert||leftovers|torrent|earthquake,icebeam,hydropump,protect|Relaxed|240,,136,40,48,44|F||||",
        "Gengar||leftovers|levitate|willowisp,thunderbolt,icepunch,explosion|Timid|168,,,,164,176|F||||",
        "Starmie||leftovers|naturalcure|hydropump,icebeam,thunderbolt,rapidspin|Timid|4,,,252,,252||,0,,,,|||"]


def p2(item="metalcoat", ability="keeneye", m4="toxic", nature="Careful", spa="", ivs=",0,,,,", rest=None):
    lead = SK.format(item=item, ability=ability, m4=m4, nature=nature, spa=spa, ivs=ivs)
    return "]".join([lead] + (rest if rest is not None else REST))


VARIANTS = {
    "lead 4th move (toxic -> drillpeck)": p2(m4="drillpeck"),
    "lead item (metalcoat -> sharpbeak)": p2(item="sharpbeak"),
    "lead SpA EVs (0 -> 252)": p2(spa="252"),
    "lead nature (Careful -> Calm: +SpD kept, Atk/SpA unused)": p2(nature="Calm"),
    "lead IVs (SpA 31 -> 0)": p2(ivs=",0,,0,,"),
    "lead ability (keeneye -> sturdy)": p2(ability="sturdy"),
    "bench slot 6 species (Starmie -> Zapdos)": p2(rest=REST[:4] + ["Zapdos||leftovers|pressure|thunderbolt,roar,rest,toxic|Calm|224,,,,252,32||,0,,,,|||"]),
    "bench slot 2 Blissey item (leftovers -> lumberry)": p2(rest=[REST[0].replace("leftovers", "lumberry")] + REST[1:]),
    "bench Tyranitar HP type (hpbug -> hpgrass IVs)": p2(rest=[REST[0], REST[1].replace("hiddenpowerbug", "hiddenpowergrass").replace(",30,30,,30,", "30,,,30,,"), *REST[2:]]),
}

if __name__ == "__main__":
    seeds = [[11, 22, 33, 44], [5, 6, 7, 8], [901, 3, 77, 12]]
    base = p2()
    for sw in seeds:
        print(f"######## seed {sw}")
        for name, var in VARIANTS.items():
            L.compare(name, [P1, base], [P1, var], sw)
        sys.stdout.flush()
