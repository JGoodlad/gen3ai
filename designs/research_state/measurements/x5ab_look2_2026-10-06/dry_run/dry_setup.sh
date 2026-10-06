#!/usr/bin/env bash
# scratch models dir (symlinks, read-only targets) for the look-2 CPU dry run
set -e
S=/home/goodlad/.cache/gen3ai/x5look2/dry
M=/home/goodlad/dev/gen3ai/models
rm -rf "$S"
mkdir -p "$S/models" "$S/ledger" "$S/plan"
for arm in fm blob; do
  for s in 1001 1002; do
    r=rb_x5ab_${arm}_s$s
    mkdir -p "$S/models/$r/checkpoints"
    for f in final_model.zip final_model.json metadata.json model_config.json; do ln -s "$M/$r/$f" "$S/models/$r/$f"; done
    for c in $(ls "$M/$r/checkpoints" | grep -E '^checkpoint_1[25][0-9]{6}_steps.(zip|json)$'); do
      ln -s "$M/$r/checkpoints/$c" "$S/models/$r/checkpoints/$c"
    done
  done
done
find "$S/models" | sort
