"""Report unique-reference count and citation placement per section."""
import io
import re

P = "manuscript/els-cas-templates/manuscript-v4.tex"
s = io.open(P, encoding="utf-8", newline="").read()

# strip body-less preamble
body = s[s.index("\\begin{document}"):]

INTRO_LABEL = "Introduction (unlabelled)"
cur = INTRO_LABEL
rows = []
order = []
for line in body.splitlines():
    m = re.match(r"\\section\{([^}]*)\}|\\subsection\{([^}]*)\}", line)
    if m:
        cur = (m.group(1) or m.group(2))
        cur = re.sub(r"\\label\{[^}]*\}", "", cur).strip()
        if cur not in order:
            order.append(cur)
    keys = []
    for grp in re.findall(r"\\cite[a-z]*\{([^}]*)\}", line):
        keys += [k.strip() for k in grp.split(",") if k.strip()]
    if keys:
        rows.append((cur, len(keys), keys))

agg = {}
for sec, n, keys in rows:
    a = agg.setdefault(sec, [0, set()])
    a[0] += n
    a[1].update(keys)

print("%-58s %6s %6s" % ("section", "cmds", "uniq"))
print("-" * 72)
for sec in [INTRO_LABEL] + [o for o in order if o in agg]:
    if sec in agg:
        print("%-58s %6d %6d" % (sec[:58], agg[sec][0], len(agg[sec][1])))
allk = set()
for _, _, keys in rows:
    allk.update(keys)
print("-" * 72)
print("TOTAL unique references cited: %d   (citation commands: %d)"
      % (len(allk), sum(n for _, n, _ in rows)))
