"""THE SOURCE REGISTER: every page a team was taken from, with its provenance as written on the page.

Provenance fields were read from the public Smogon pages on 2026-10-18 (through the WebFetch reader,
since the forum serves a JavaScript browser check to plain HTTP clients — README FINDING F1). The team
TEXT is never taken from that reader except for the rows marked ``transcribed`` (README FINDING F2):
it is the pokepast.es JSON body or the pastebin.com page's own source listing.

Tiers (the brief's): A = official sample teams / tournament-used / team-archive entries (including the
team dumps and team analyses that the ADV Beginner's Lounge resource post lists); B = RMTs or
competition entries rated well by experienced posters / judges; C = other.
"""

SAMPLE = {"thread_url": "https://www.smogon.com/forums/threads/adv-ou-sample-teams.3687813/",
          "thread_title": "Resource - ADV OU Sample Teams"}
LOUNGE = {"thread_url": "https://www.smogon.com/forums/threads/adv-beginners-lounge-rules-faq-resources-question-answer.3687817/",
          "thread_title": "ADV Beginner's Lounge - Rules/FAQ, Resources, Question & Answer"}
AGES = {"thread_url": "https://www.smogon.com/forums/threads/adv-teams-through-the-ages.3687203/",
        "thread_title": "OU ADV Teams Through The Ages"}
TBC = {"thread_url": "https://www.smogon.com/forums/threads/adv-ou-teambuilding-competition.3770678/",
       "thread_title": "OU - ADV OU Teambuilding Competition"}

# (pokepaste id, team name as listed, credited builder as listed)
_SAMPLE_P1 = [
    ("f6229d2c867e21d6", "Big 5 + Starmie (Beerlover)", "UD"),
    ("a04c29cf769e9a11", "Forretress Gengar Dug", "M Dragon"),
    ("82e97fe2bfbb6f69", "Gengar + Jirachi \"Superman\" TSS", "ADV Community"),
    ("9f27f5d3e34021a7", "Moltres \"Superman\" TSS", "Golden Sun"),
    ("b89e1e37caa40e6a", "SkarmMag + Aerodactyl TSS", "ABR"),
    ("e11829f0561ef5a9", "MixMence Claydol TSS", "ADV Community"),
    ("9909f2e98e981ccc", "Zapdos MixMence Forre TSS", "ADV Community"),
    ("9283210847f806ee", "Special Wall-less TSS \"El clasico\"", "Hclat"),
    ("dbf81d8ecae51c39", "Jolteon Starmie Spikes Stack", "Kerts"),
    ("d4e74946b54f1a4b", "Zapdos + AeroBi Spikes Offense", "Shitrock enjoyer"),
    ("9eb3abdc52876a63", "DDTar + Starmie Spikes Offense", "Shitrock enjoyer"),
    ("ce35b7368c3d692e", "Regice Spikes Offense", "Jirachee"),
    ("90b94599967c6b77", "Charizard Spikes Offense", "giraffefromholland"),
    ("6d56d06b9fca508e", "Tyranitar + MiloDol TSS", "Zokuru"),
    ("f7ba5702fe856292", "Zapdos + CuneDol", "ADV Community"),
    ("0fa105ae42dc85ec", "Triple Natural Cure", "ABR"),
    ("8cdc78b2d46f0515", "Suicune + MiloDol (\"Vir5in\")", "Kerts"),
    ("04e417ef9822abe9", "Curse RestLax ForreDol", "ABR"),
    ("460b8c99ee366f6c", "Smeargle Offense", "ADV Community"),
    ("63eda9d8d491d6a4", "Blue Offense", "Noitu"),
    ("f5a4f4f0f5dc49ce", "Medicham Cune Offense", "Mana"),
    ("53f215cb45b7d729", "Aero SD Pass", "z0mOG"),
    ("eccfe630ec08de27", "ZapDug", "ADV Community"),
    ("023a2d47648b85e6", "Jynx Special Offense", "Hclat"),
    ("61590463ee85d456", "Quad Band", "M Dragon"),
    ("7036a0a1dcb59a19", "CB Mence MagDol", "BKC"),
    ("a3ca232f030ef625", "Charizard Houdini CMPass", "SEA"),
    ("e541f7be8713393c", "CBMeta CroCune Skarm beatdown", "ADV Community"),
    ("710d8d529538ff90", "MixZap Houdini DDTar MixOff", "ADV Community"),
    ("e0d97b0ed592889d", "WishJira Skarm Beatdown", "Astamatitos"),
    ("8e768980fc8f3b5f", "Raikou Celebi Slop", "Shitrock enjoyer and giraffefromholland"),
    ("9d5f845869e899ee", "Yama BandMence", "ADV Community"),
]
_SAMPLE_P2 = [
    ("26c7b11a3462fd02", "Molt Superman (2018)", "Golden Sun"),
    ("0eb7938532db7b60", "SkarmMag + Aerodactyl TSS (2015)", "BKC, updated by Astamatitos"),
    ("558438a7a8310b94", "Moltres TSS (2018-2024)", "eden"),
    ("49f23151c6b611d7", "MixMence Aero Forre TSS (2021)", "Star"),
    ("1fc6068e9111e80b", "MixMence + BKCTar Spikes Offense (2023)", "Shitrock enjoyer"),
    ("cadefacc14440fc0", "Double Status Zapdos + Dugtrio + CM Celebi Balance (2015-2022)", "Astamatitos"),
    ("8e1401a81e0be602", "Milotic Claydol TSS (2015-2024)", "Zokuru"),
    ("ad85129d1352b49c", "Hariyama Marowak SpinMie Balance (2020)", "Star"),
    ("2ad2fcb7e3f831c8", "SubZap OffYama SpinMie Balance (2021)", "TDK, submitted by Star"),
    ("f5e3f3222302ba96", "Blue Offense (2015)", "Noitu"),
    ("e691bb4d11ddf4e3", "Mixed Offense (2015-2019)", "thelinearcurve"),
    ("af8824d77b9678c4", "ZapDug Offense (2015-2024)", "eden"),
    ("aa87ad1b92137b49", "Smeargle Vaporeon offense (2018)", "eden"),
    ("50587d629585e1a4", "Smeargle Zapdos offense (2018)", "eden"),
    ("108a2bcaa27727da", "Charizard CM Pass (2024)", "SEA"),
    ("4972596fc1ed58c2", "CM Spam (2013)", "dekzeh"),
    ("1214a19de0fe9ba7", "Magneton Special Spam (2020)", "vapicuno"),
    ("ee15fa5b964f919e", "Medicham Mag Offense (2023)", "Mana"),
    ("ea74262205422c17", "CBMeta Mag SubPassBi DDSpam (2021)", "vapicuno"),
    ("0181d76e2f597cc9", "MagDol + Double Set-up (2015)", "CZ."),
    ("252fbc75fa28087d", "Aero SD Pass (2019)", "z0mOG"),
    ("f09ff66281cee40e", "Milotic SD Pass (2020)", "Teclis"),
]
_AGES = [  # M Dragon's history posts (Jul 4, 2021); builder as the post names it, "" when not legible
    ("448f1b81ebee106e", "MT offensive team (Italian metagame)", ""),
    ("f46695be768cf176", "Fra Stall team (Italian metagame)", ""),
    ("26e2635e96472fce", "Carl Classic (Italian metagame)", ""),
    ("a1020c590086af16", "Revengezard", "Sharingan"),
    ("ec23705ddd7f14a1", "Did ya know RSE aer Fat", "g_f"),
    ("bab822d3a6a24816", "PursuiTar", "g_f"),
    ("77b9107ec90cdb6f", "Stall Breaker 2009", ""),
    ("fca1f5218e2381eb", "Ivan Drago", "Zetsu91"),
    ("ba64de736ca6e345", "Team by Dinozzo", "Dinozzo"),
    ("a375da3005e8b7d0", "Vertigo's CM Bliss TSS (German metagame)", "Vertigo"),
    ("35d237136a776484", "Peter Pan's main team (German metagame)", "Peter Pan"),
    ("c6dcfd9407abf63c", "SD Celebi team (German metagame)", ""),
    ("5340ec33b6a73295", "CM BP Celebi (German metagame)", ""),
    ("8799fae8dd447012", "Balanced Dugtrio team (German metagame)", ""),
    ("a6217617cb089a26", "Two example teams (French metagame)", ""),
    ("b1c6278b15d9fe9d", "physical offense team", ""),
    ("877aa376ea86c5c0", "double dogs / Flygon TSS", ""),
    ("17ae394678500b9a", "Steelix TSS", ""),
    ("d64ad6336ce9b754", "Suicune Spikes", "make"),
    ("418b06b704dc7d35", "Offensive Starmie TSS", ""),
    ("0e0e1fef3c573eec", "Forry Skarm + Double Special Walls", ""),
    ("821b3c9b4ad65418", "Steelix anti Spikes", ""),
    ("cb36ecfd36a833b0", "Magneton Physical Offense", ""),
    ("59c8bd061142c5dd", "Modest Zapdos", ""),
    ("b3007308b34e6fcb", "Zapdos + Dugtrio + Gengar", ""),
    ("7b1f4039ba774246", "Majin Quadruple CM Balance", ""),
    ("191f2b915ab31468", "balanced Mixmence Aero", ""),
    ("d30c84b93bbcf0f8", "Aero offensive", ""),
    ("33e5289413855275", "Undisputed TTar + Flygon + Aero", "Undisputed"),
    ("5c32d4db5d666eaa", "SD Heracross Offense", ""),
    ("9f896614e5de2646", "Milotic Flygon Balance", ""),
]
_TBC = [  # round winners, the first post's Hall of Fame
    ("aef2f2d61002a81f", "R1 Medicham: Bulk Up Pass Medicham + Triple Trappers (winner)", "HeyYa!_Forte"),
    ("558007e4ed74293c", "R2 Flygon no Skarmory: Forre OffGon + Charizard (winner)", "Johnald"),
    ("fea738831a609237", "R3 no Spikes/Sand/Trappers: MACONHA FESTA (winner)", "Shitrock enjoyer"),
    ("01a52fe417cfef54", "R4 Raikou: Kuno Reverse Card (winner)", "Spreek"),
    ("4fd59a92b5db10a2", "R5 Spikeless no Suicune no DD Salamence: Nenriki (winner)", "mielke"),
    ("69527e1d2cc2e98b", "R6 Statpass no Statpass Celebi: Thing (winner)", "L3W"),
    ("8abd56103f32fe49", "R7 UU-or-below Water type: Lapras Loom Mixoff (winner)", "Johnald"),
]


def pokepaste_sources():
    out = []
    for pid, name, by in _SAMPLE_P1:
        out.append(dict(paste_id=pid, team_name=name, credited=by, post_author="mielke",
                        post_date="2021-07-16", tier="A", category="sample_team", **SAMPLE))
    for pid, name, by in _SAMPLE_P2:
        out.append(dict(paste_id=pid, team_name=name, credited=by, post_author="vapicuno",
                        post_date="2021-07-16", tier="A", category="sample_team_archived", **SAMPLE))
    for pid, name, by in _AGES:
        out.append(dict(paste_id=pid, team_name=name, credited=by, post_author="M Dragon",
                        post_date="2021-07-04", tier="A", category="team_archive_history", **AGES))
    out.append(dict(paste_id="c280810585fcc2b4", team_name="Loki-style Gengar Zapdos Flygon Skarmory Swampert",
                    credited="Loki (as recalled)", post_author="Taylor", post_date="2022-06-05",
                    tier="A", category="team_archive_history", **AGES))
    for pid, name, by in _TBC:
        out.append(dict(paste_id=pid, team_name=name, credited=by, post_author=by, post_date="2025-09..2026",
                        tier="B", category="teambuilding_competition_winner", **TBC))
    out += [
        dict(paste_id="01cd432a1779a56c", team_name="Indestructible (Camerupt offense)", credited="EG Jhonx~",
             post_author="EG Jhonx~", post_date="2021-12-13", tier="A", category="resource_listed_team_analysis",
             thread_url="https://www.smogon.com/forums/threads/indestructible.3694811/",
             thread_title="Indestructible"),
        dict(paste_id="be776252fd563133", team_name="Timeless Adventure (Milotic SD Pass, Snorlax version)",
             credited="Teclis", post_author="Teclis", post_date="2020-10-16", tier="A",
             category="resource_listed_team_analysis",
             thread_url="https://www.smogon.com/forums/threads/timeless-adventure-a-1k-rmt.3671930/",
             thread_title="Timeless Adventure - a 1k RMT"),
        dict(paste_id="9d85a630a2f9c12c", team_name="Dragon Dance Tyranitar Offense", credited="McMeghan",
             post_author="RMT Staff", post_date="2019-08-14", tier="A", category="resource_listed_team_analysis",
             thread_url="https://www.smogon.com/forums/threads/dragon-dance-tyranitar-offense-by-mcmeghan.3653511/",
             thread_title="Gen 3 Dragon Dance Tyranitar Offense by McMeghan"),
    ]
    return out


# Multi-team pastebin dumps linked from the Sample Teams thread (post 2) and the Beginner's Lounge.
# Fetched as the normal paste page (robots.txt disallows /raw/ — README FINDING F3).
PASTEBIN_DUMPS = [
    ("6xCb1crp", "2021 thelinearcurve", "thelinearcurve"),
    ("tUytkX29", "2020 thelinearcurve (all)", "thelinearcurve"),
    ("QbwUxw0P", "2020 thelinearcurve (more viable/user-friendly)", "thelinearcurve"),
    ("jyFh4AF3", "2019 thelinearcurve", "thelinearcurve"),
    ("i7HzJQmC", "2017 Ibidem", "Ibidem"),
]
GOLDEN_SUN = {"thread_url": "https://www.smogon.com/forums/threads/team-dump.3576354/",
              "thread_title": "OU Team dump", "post_author": "Golden Sun", "post_date": "2016-07-03"}
GOLDEN_SUN_PASTES = ["7kNkVWxP", "MfEzJsH3", "qNspHuRe", "JG307Jin", "fmDC1uNc", "UpruJpjr"]


def pastebin_sources():
    out = []
    for pid, name, by in PASTEBIN_DUMPS:
        out.append(dict(paste_id=pid, team_name=name, credited=by, post_author="vapicuno",
                        post_date="2021-07-16", tier="A", category="curated_team_dump", multi=True, **SAMPLE))
    for pid in GOLDEN_SUN_PASTES:
        out.append(dict(paste_id=pid, team_name="", credited="Golden Sun", tier="A",
                        category="curated_team_dump", multi=False, **GOLDEN_SUN))
    return out
