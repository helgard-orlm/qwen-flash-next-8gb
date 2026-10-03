#!/bin/bash
# One-time preparation of the working directory QW_DIR from a Hugging Face snapshot in QW_ORIG.
#   1) download:  huggingface-cli download nvidia/Qwen3.8-Flash-Next-NVFP4 \
#                   --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 --local-dir "$QW_ORIG"
#   2) QW_ORIG=... QW_DIR=... ./setup.sh
# QW_DIR must be on a fast NVMe (experts are read with O_DIRECT, the n-gram table by random rows).
# The non-expert weights (nonexpert.safetensors) are extracted from QW_ORIG on the first engine start.
set -eu
: "${QW_ORIG:?set QW_ORIG}" "${QW_DIR:?set QW_DIR}"
mkdir -p "$QW_DIR"
for f in config.json generation_config.json hf_quant_config.json chat_template.jinja merges.txt vocab.json \
         tokenizer.json tokenizer_config.json model.safetensors.index.json model-fp8-mtp-ple.safetensors; do
  cp --preserve=timestamps "$QW_ORIG/$f" "$QW_DIR/$f"; echo "copied $f"
done
python "$(dirname "$0")/repack_experts.py"          # -> experts.bin (63.3 GiB) + experts_scal.npy
python "$(dirname "$0")/tools/check_repack.py"      # random experts: repacked block == original bytes
