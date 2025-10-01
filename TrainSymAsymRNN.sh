#!/usr/bin/env bash
set -euo pipefail

export WHH_TYPE="identity"        # or baseline, cycshift, ...
export WHH_NORM="frobenius"     # frobenius|spectral|variance|none
export INPUT="asym1"           # suffix in your file name (e.g., 1, asym1, etc.)

# For baseline we still write a 'none' norm folder so paths are uniform
NORM_DIR="${WHH_NORM}"
if [[ "${WHH_TYPE}" == "baseline" ]]; then
  NORM_DIR="none"
fi

# New base includes <whh_type>/<whh_norm>/<input>
BASE="SymAsymRNN/N100T100/${WHH_TYPE}/${NORM_DIR}/${INPUT}"

echo "[plan] saving runs under: ${BASE}/multiruns/run_XX/"
echo "[plan] encoding file: data/Ns100_SeqN100/encodings/Ns100_SeqN100_${INPUT}.pth.tar"

for i in $(printf "%02d\n" {0..9}); do
  RUN_DIR="${BASE}/multiruns/run_${i}"
  HW_DIR="${RUN_DIR}/hidden-weights"     # keep plural; it’s what Main_s4.py writes
  mkdir -p "${HW_DIR}"

  nohup python Main_s4.py \
    --input "data/Ns100_SeqN100/encodings/Ns100_SeqN100_${INPUT}.pth.tar" \
    --net SymAsymRNN \
    --batch-size 1 \
    --pred 1 --fixi 1 --hidden-n 100 \
    --seed "${i}" \
    --epochs 20000 \
    --whh_type "${WHH_TYPE}" --whh_norm "${WHH_NORM}" \
    --output_dir "${HW_DIR}" \
    --savename  "${RUN_DIR}/Ns100_SeqN100_predloss_full" \
    > "${RUN_DIR}/train.out" 2>&1
done

echo "Launched 10 runs under ${BASE}/multiruns/"





# ONE RUN
WHH_TYPE="baseline"
WHH_NORM="none"
INPUT="asym1"
i="02"

BASE="SymAsymRNN/N100T100/${WHH_TYPE}"
[[ "${WHH_TYPE}" != "baseline" ]] && BASE="${BASE}/${WHH_NORM}"
RUN_DIR="${BASE}/multiruns/run_${i}"
mkdir -p "${RUN_DIR}/hidden-weights"

python Main_s4.py \
  --input "data/Ns100_SeqN100/encodings/Ns100_SeqN100_${INPUT}.pth.tar" \
  --net SymAsymRNN \
  --batch-size 1 \
  --pred 1 --fixi 1 --hidden-n 100 \
  --seed "${i}" \
  --epochs 10000 \
  --early_stop 1 \
  --whh_type "${WHH_TYPE}" --whh_norm "${WHH_NORM}" \
  --output_dir "${RUN_DIR}/hidden-weights" \
  --savename  "${RUN_DIR}/Ns100_SeqN100_predloss_full"
