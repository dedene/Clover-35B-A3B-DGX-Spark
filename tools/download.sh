#!/usr/bin/env bash
# Fetch the pinned Clover-35B-A3B-Preview checkpoint (BF16 safetensors, 69 GB) into /models. Resumable: the hf CLI
# skips files it already has and checks each LFS file's sha256. A later start is a no-op once the revision stamp
# matches.
set -euo pipefail

model_id=${MODEL_ID:-Diffuspace/Clover-35B-A3B-Preview}
revision=${MODEL_REVISION:-4b0402b65e523d0183a14c431955438714cb16ff}
# Keep this directory name in step with tools/serve.sh.
dest=/models/Clover-35B-A3B-Preview
stamp=/models/.Clover-35B-A3B-Preview.revision

if [[ -f $stamp && $(<"$stamp") == "$model_id $revision" && -f $dest/model.safetensors.index.json ]]; then
  echo "present $model_id@$revision"
  exit 0
fi
rm -f "$stamp"
echo "downloading $model_id@$revision"
hf download "$model_id" --revision "$revision" --local-dir "$dest"
printf '%s %s\n' "$model_id" "$revision" >"$stamp"
echo "verified $model_id@$revision"
exit 0
