"""Print the EFFECTIVE tau_solve / trunk_precision / behaviour_check / fork-arch verdict a validated argv resolves to,
through the TRAINER's own resolution (main.train.fork_arch.resolve_like_the_trainer: the trainer's parser + resolve_config,
which is where `_resolve("tau_solve", ...)` runs), and the trainer's own architecture check. Read-only, CPU."""
import shlex, sys
from main.checkargs import resolve_against_parent
from main.train.fork_arch import resolve_like_the_trainer, fork_arch_verdict
for path in sys.argv[1:]:
    argv = shlex.split(open(path).read())
    model = argv[argv.index("--model") + 1]
    ns = resolve_like_the_trainer(argv, model)
    fa = fork_arch_verdict(argv, model)
    tag = path.rsplit("argv_", 1)[-1][:-4]
    ok = fa is not None and fa.checked and not fa.refused
    print(f"{tag}: tau_solve={ns.tau_solve} trunk_precision={ns.trunk_precision} "
          f"behaviour_check={getattr(ns,'behaviour_check',None)} fork_arch={'OK(checked)' if ok else fa}")
