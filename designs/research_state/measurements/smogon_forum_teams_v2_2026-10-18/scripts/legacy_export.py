"""PRE-SHOWDOWN EXPORT SPELLING -> today's export spelling (format only; every value is kept).

Written by the old_rmt crawler (2026-10-18) to compare two transcriptions of 2007-2015 posts, which
use ``Trait:``, ``SAtk/SDef/Spd``, ``Nature: X`` and one-line ``Moves: a, b, c``; adopted here as the
canonical form for (a) ``crawl.py inline-check`` and (b) the v2 build's pre-pass on TRANSCRIBED rows.
Negative controls by its author: one changed move or EV value still disagrees. It drops ONLY an
``IVs:`` line whose six values are all 31 (the default; a reader often invents one).
"""
import re
STAT={'hp':'HP','atk':'Atk','def':'Def','satk':'SpA','spa':'SpA','spatk':'SpA','sp.atk':'SpA','spdef':'SpD','sdef':'SpD','spd':None,'sp.def':'SpD','spe':'Spe','spd.':'Spe','speed':'Spe'}
def stat(s):
    k=s.lower().replace(' ','')
    if k=='spd': return 'SPD?'
    return STAT.get(k,s)
def canon(t):
    out=[];
    for ln in t.splitlines():
        s=ln.strip()
        if not s: out.append(''); continue
        m=re.match(r'(?i)^(trait|ability):\s*(.*)',s)
        if m: out.append('Ability: '+m.group(2)); continue
        m=re.match(r'(?i)^nature:\s*([A-Za-z]+)',s) or re.match(r'^([A-Za-z]+) Nature',s)
        if m: out.append(m.group(1).capitalize()+' Nature'); continue
        m=re.match(r'(?i)^(evs|ivs):\s*(.*)',s)
        if m:
            parts=[p.strip() for p in m.group(2).split('/')]
            if m.group(1).lower()=='ivs':
                vals=[re.match(r'\d+',p).group(0) if re.match(r'\d+',p) else p for p in parts]
                if all(v=='31' for v in vals): continue
            sp=[]
            for p in parts:
                mm=re.match(r'(\d+)\s*(.*)',p)
                if not mm: sp.append(p); continue
                st=mm.group(2).strip()
                k=st.lower().replace(' ','')
                if st=='SpD': k='spdef'
                elif st=='Spd': k='speed'
                st={'hp':'HP','atk':'Atk','def':'Def','satk':'SpA','spa':'SpA','spatk':'SpA','sp.atk':'SpA','sdef':'SpD','spdef':'SpD','sp.def':'SpD','spe':'Spe','speed':'Spe'}.get(k,st)
                sp.append(f'{mm.group(1)} {st}')
            out.append(m.group(1).upper()[:2]+'s: '+' / '.join(sp)); continue
        m=re.match(r'(?i)^moves:\s*(.*)',s)
        if m:
            for mv in m.group(1).split(','): out.append('- '+fixhp(mv.strip()))
            continue
        if s.startswith('-'): out.append('- '+fixhp(s[1:].strip())); continue
        out.append(s)
    return '\n'.join(out)+'\n'
def fixhp(mv):
    m=re.match(r'(?i)^hidden power\s*\[?([A-Za-z]+)\]?.*',mv)
    return f'Hidden Power [{m.group(1).capitalize()}]' if m else mv
