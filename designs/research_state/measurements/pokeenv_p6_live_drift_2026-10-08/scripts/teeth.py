from main.live.reader import LiveReader, ReaderRefusal

PREAMBLE = ["|player|p1|p1user||", "|player|p2|p2user||", "|teamsize|p1|6", "|teamsize|p2|6", "|gametype|singles",
            "|gen|3", "|start", "|switch|p1a: Zappy|Zapdos, L100|100/100", "|switch|p2a: Snorlax|Snorlax, L100, M|100/100",
            "|turn|1", "|move|p2a: Snorlax|Tackle|p1a: Zappy"]
for text in ["|-start|p2a: Snorlax|bogusvolatile", "|-activate|p2a: Snorlax|move: Bogus Move",
             "|-singleturn|p2a: Snorlax|Bogus", "|-start|p2a: Snorlax|Substitute", "|-activate|p2a: Snorlax|move: Heal Bell",
             "|-start|p2a: Snorlax|move: Yawn"]:
    r = LiveReader()
    try:
        r.open("p1", "p1user", None)
        r.feed(PREAMBLE)
        r.feed([text])
        print(text, "->", r.probe())
    except ReaderRefusal as e:
        print(text, "-> REFUSED", e.kind, str(e)[:160])
    r.close()
