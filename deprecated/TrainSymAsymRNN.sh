#!/usr/bin/env bash
set -euo pipefail

export WHH_TYPE="learned"
export WHH_NORM="frobenius"
export INPUT="asym1"

# NEW: choose initial lambda
LAMBDA0="0.75"   # e.g. 0.25, 0.50, 0.75

# format "lambda0p50" from "0.50"
LAM_PCT="$(awk -v x="$LAMBDA0" 'BEGIN{printf "%d", int(x*100 + 0.5)}')"
LAM_TAG="lambda0p$(printf '%02d' "$LAM_PCT")"

NORM_DIR="${WHH_NORM}"
if [[ "${WHH_TYPE}" == "baseline" ]]; then
  NORM_DIR="none"
fi

BASE="SymAsymRNN/N100T100/${LAM_TAG}/${WHH_TYPE}/${NORM_DIR}/${INPUT}"
echo "[plan] saving runs under: ${BASE}/multiruns/run_XX/"
echo "[plan] encoding file: data/Ns100_SeqN100/encodings/Ns100_SeqN100_${INPUT}.pth.tar"

for i in $(printf "%02d\n" {0..2}); do
  RUN_DIR="${BASE}/multiruns/run_${i}"
  HW_DIR="${RUN_DIR}/hidden-weights"
  mkdir -p "${HW_DIR}"

  nohup python Main_s4.py \
    --input "data/Ns100_SeqN100/encodings/Ns100_SeqN100_${INPUT}.pth.tar" \
    --net SymAsymRNN \
    --batch-size 1 \
    --pred 1 --fixi 1 --hidden-n 100 \
    --seed "${i}" \
    --epochs 30000 \
    --whh_type "${WHH_TYPE}" --whh_norm "${WHH_NORM}" \
    --lambda0 "${LAMBDA0}" \
    --output_dir "${HW_DIR}" \
    --savename  "${RUN_DIR}/Ns100_SeqN100_predloss_full" \
    > "${RUN_DIR}/train.out" 2>&1
done

echo "Launched 3 runs under ${BASE}/multiruns/"





