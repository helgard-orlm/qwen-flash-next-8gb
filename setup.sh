#!/bin/bash
# One-time preparation of the working directory QW_DIR.
#
#   QW_ORIG=/path/to/hf-snapshot QW_DIR=/path/on/nvme/qwen38 ./setup.sh
#
# QW_ORIG  the Hugging Face snapshot. If it has no config.json, it is downloaded there (133 GB, pinned revision).
#          May be on a slow disk (HDD): it is read once, sequentially.
# QW_DIR   the working directory. Must be on a fast NVMe: experts are read with O_DIRECT at run time,
#          the n-gram table by random rows.
#
# Same filesystem for both (one disk): the PLE table and configs are hard-linked, not copied.
# After setup QW_ORIG is not needed any more; the script says how much deleting it frees.
set -eu
: "${QW_ORIG:?set QW_ORIG}" "${QW_DIR:?set QW_DIR}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=nvidia/Qwen3.8-Flash-Next-NVFP4
REV=fc694b54fb0174e0913e6adf86691ef85a4ead47
SNAPSHOT=132766433236        # bytes of the snapshot at REV
EXPERTS=67947724800          # experts.bin
NONEXP=9895548336            # nonexpert.safetensors
PLE=53717551730              # model-fp8-mtp-ple.safetensors
SMALL="config.json generation_config.json hf_quant_config.json chat_template.jinja merges.txt vocab.json
       tokenizer.json tokenizer_config.json model.safetensors.index.json"

mkdir -p "$QW_ORIG" "$QW_DIR"
gb() { echo $(( $1 / 1000000000 )); }
free_on() { df -B1 --output=avail "$1" | tail -1; }
same_fs() { [ "$(stat -c %d "$QW_ORIG")" = "$(stat -c %d "$QW_DIR")" ]; }

# ---- 1. space check before doing anything
download=0; [ -f "$QW_ORIG/config.json" ] || download=1
need_dir=$(( EXPERTS + NONEXP ))
same_fs || need_dir=$(( need_dir + PLE ))
if same_fs; then
  need=$(( need_dir + download * SNAPSHOT ))
  [ "$(free_on "$QW_DIR")" -ge "$need" ] || { echo "NOT ENOUGH SPACE: need $(gb $need) GB on the disk of $QW_DIR (one disk mode), free $(gb "$(free_on "$QW_DIR")") GB"; exit 1; }
  echo "one disk: QW_ORIG and QW_DIR on the same filesystem, need $(gb $need) GB"
else
  [ "$(free_on "$QW_DIR")" -ge "$need_dir" ] || { echo "NOT ENOUGH SPACE: need $(gb $need_dir) GB for $QW_DIR, free $(gb "$(free_on "$QW_DIR")") GB"; exit 1; }
  if [ $download = 1 ] && [ "$(free_on "$QW_ORIG")" -lt $SNAPSHOT ]; then
    echo "NOT ENOUGH SPACE: need $(gb $SNAPSHOT) GB for the download in $QW_ORIG"; exit 1; fi
  echo "two disks: need $(gb $need_dir) GB in QW_DIR$( [ $download = 1 ] && echo " + $(gb $SNAPSHOT) GB download in QW_ORIG")"
fi

# ---- 2. download the snapshot if it is not there
if [ $download = 1 ]; then
  echo "downloading $REPO@$REV into $QW_ORIG (133 GB) ..."
  if command -v hf >/dev/null; then hf download "$REPO" --revision "$REV" --local-dir "$QW_ORIG"
  else huggingface-cli download "$REPO" --revision "$REV" --local-dir "$QW_ORIG"; fi
fi

# ---- 3. configs + PLE table: hard link on one disk, copy otherwise
for f in $SMALL model-fp8-mtp-ple.safetensors; do
  [ -e "$QW_DIR/$f" ] && { echo "exists $f"; continue; }
  if same_fs; then ln "$QW_ORIG/$f" "$QW_DIR/$f"; echo "linked $f"
  else cp --preserve=timestamps "$QW_ORIG/$f" "$QW_DIR/$f"; echo "copied $f"; fi
done

# ---- 4. experts: repack + check
if [ -f "$QW_DIR/experts.bin" ] && [ "$(stat -c %s "$QW_DIR/experts.bin")" = $EXPERTS ] && [ -f "$QW_DIR/experts_scal.npy" ]; then
  echo "exists experts.bin"
else
  python "$HERE/repack_experts.py"
fi
python "$HERE/tools/check_repack.py"

# ---- 5. non-expert weights (otherwise the engine would need QW_ORIG on its first start)
if [ -f "$QW_DIR/nonexpert.safetensors" ]; then echo "exists nonexpert.safetensors"
else (cd "$HERE" && python -c "import qwen_engine as QE; print('nonexpert tensors:', QE.extract_nonexpert())"); fi

# ---- 6. what QW_ORIG is still good for
if same_fs; then
  echo "SETUP_DONE. QW_ORIG is not needed any more: deleting it frees ~$(gb $(( SNAPSHOT - PLE ))) GB (the PLE table stays, hard-linked)."
else
  echo "SETUP_DONE. QW_ORIG is not needed any more: deleting it frees $(gb $SNAPSHOT) GB on its disk."
fi
