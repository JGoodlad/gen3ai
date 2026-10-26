"""Build packed edge-case teams (12 fields each: name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|misc)."""
def mon(species, item, ability, moves, nature="", evs="", ivs="", level="", misc="", gender=""):
    return "|".join(["", species, item, ability, ",".join(moves), nature, evs, gender, ivs, "", level, misc])
T = []
T.append("]".join([   # Ditto (Transform-only, 1 move) + fillers
    mon("Ditto", "leftovers", "limber", ["transform"], "Relaxed", "252,,252,,4,"),
    mon("Skarmory", "leftovers", "keeneye", ["spikes", "roar", "protect", "drillpeck"], "Careful", "252,,,,252,4"),
    mon("Blissey", "leftovers", "naturalcure", ["thunderbolt", "calmmind", "icebeam", "softboiled"], "Modest", ",,252,228,,28", ",0,,,,"),
    mon("Metagross", "leftovers", "clearbody", ["explosion", "protect", "meteormash", "earthquake"], "Adamant", "252,56,140,,,60"),
    mon("Swampert", "leftovers", "torrent", ["earthquake", "icebeam", "protect", "refresh"], "Relaxed", "240,,136,40,48,44"),
    mon("Gengar", "leftovers", "levitate", ["willowisp", "thunderbolt", "icepunch", "explosion"], "Timid", "168,,,,164,176")]))
T.append("]".join([   # odd EVs / IVs / missing nature / neutral nature / bare HP / hpType misc / few moves
    mon("Starmie", "leftovers", "naturalcure", ["surf"], "Timid", "4,,,252,,252", ",0,,,,"),
    mon("Snorlax", "leftovers", "thickfat", ["bodyslam", "curse"], "Careful", "255,85,1,,167,", "31,0,31,31,31,0"),
    mon("Zapdos", "leftovers", "pressure", ["thunderbolt", "hiddenpower", "substitute", "toxic"], "", ",,,252,4,252"),
    mon("Jirachi", "leftovers", "serenegrace", ["wish", "protect", "bodyslam", "hiddenpower"], "Hardy", "252,,,,252,4", "30,2,31,30,31,30"),
    mon("Tyranitar", "leftovers", "sandstream", ["rockslide", "earthquake", "pursuit", "hiddenpower"], "Adamant", "248,196,12,,52,", "", "", ",Bug"),
    mon("Celebi", "leftovers", "naturalcure", ["leechseed", "recover", "psychic", "batonpass"], "Bold", "252,,252,,4,")]))
T.append("]".join([   # formes / special species
    mon("Castform", "leftovers", "forecast", ["sunnyday", "weatherball", "thunderbolt", "icebeam"], "Modest", ",,,252,4,252"),
    mon("Unown", "twistedspoon", "levitate", ["hiddenpower"], "Modest", ",,,252,4,252", "31,30,31,30,31,31"),
    mon("Smeargle", "lumberry", "owntempo", ["spore", "spikes", "batonpass", "substitute"], "Jolly", "252,,4,,,252"),
    mon("Shedinja", "lumberry", "wonderguard", ["shadowball", "silverwind", "hiddenpower", "toxic"], "Adamant", ",252,,,4,252", "31,31,31,30,31,30"),
    mon("Deoxys-Speed", "leftovers", "pressure", ["spikes", "psychoboost", "recover", "taunt"], "Timid", "252,,,4,,252"),
    mon("Medicham", "choiceband", "purepower", ["focuspunch", "brickbreak", "rockslide", "shadowball"], "Adamant", "4,252,,,,252")]))
T.append("]".join([   # duplicate species (Species Clause violation)
    mon("Skarmory", "leftovers", "keeneye", ["spikes", "roar", "protect", "drillpeck"], "Careful", "252,,,,252,4"),
    mon("Skarmory", "leftovers", "keeneye", ["whirlwind", "toxic", "protect", "steelwing"], "Impish", "252,,252,,4,"),
    mon("Blissey", "leftovers", "naturalcure", ["seismictoss", "toxic", "aromatherapy", "softboiled"], "Bold", "252,,252,,4,"),
    mon("Metagross", "choiceband", "clearbody", ["explosion", "rockslide", "meteormash", "earthquake"], "Adamant", "252,252,,,,4"),
    mon("Swampert", "leftovers", "torrent", ["earthquake", "icebeam", "protect", "refresh"], "Relaxed", "240,,136,40,48,44"),
    mon("Gengar", "leftovers", "levitate", ["willowisp", "thunderbolt", "icepunch", "explosion"], "Timid", "168,,,,164,176")]))
import sys
out = T + [l.strip() for l in open(sys.argv[1]).readlines()[:2]]
which = sys.argv[3].split(",") if len(sys.argv) > 3 else None
if which:
    out = [T[int(i)] for i in which] + [l.strip() for l in open(sys.argv[1]).readlines()[:2]]
open(sys.argv[2], "w").write("\n".join(out) + "\n")
