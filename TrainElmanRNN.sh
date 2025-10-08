#python Main_clean.py --input data/Ns200_SeqN100_1.pth.tar --ae 1 --fixi 2 #--fixo 2 --pred 1 --rnn_act relu --ac_output sigmoid --epoch 100000 --savename #Elman_SGD/Remap_predloss/N200T100_relu_fixio/pred_relu

#!/usr/bin/env bash
# Minimal launcher for Main_clean.py (Elman RNN trainer)

# ===== USER PARAMS =====
EPOCHS=20000
LR=0.01
GPU=1
HIDDEN_N=100
INPUT_PATH="./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar"

# Hidden-weight init (use 'none' to skip overriding)
WHH_TYPE="none"               # one of: none, cent, cent-cyc, shifted, shifted-cyc, identity, shift, shift-cyc
WHH_NORM="frobenius"          # {frobenius, raw} (ignored if WHH_TYPE=none)
ALPHA=0.9                     # required only for shifted*/shift* (ignored if WHH_TYPE=none)

SAVE_NAME="runs/${WHH_TYPE}_${WHH_NORM}_alpha${ALPHA}"

# ===== BUILD ARGS =====
ARGS=(
  --epochs ${EPOCHS}
  --lr ${LR}
  --gpu ${GPU}
  --hidden-n ${HIDDEN_N}
  --input "${INPUT_PATH}"
  --savename "${SAVE_NAME}"
  --pred 1
  --print-freq 1000
  --seed 1337
)

if [[ "${WHH_TYPE}" == "none" ]]; then
  echo "[whh] Using standard random init (no override)."
  # still pass the type as 'none' so the script prints a clear message
  ARGS+=( --whh_type none )
else
  ARGS+=( --whh_type "${WHH_TYPE}" --whh_norm "${WHH_NORM}" )
  # only add --alpha for the shifted/shift families
  if [[ "${WHH_TYPE}" == "shifted" || "${WHH_TYPE}" == "shifted-cyc" || "${WHH_TYPE}" == "shift" || "${WHH_TYPE}" == "shift-cyc" ]]; then
    ARGS+=( --alpha "${ALPHA}" )
  fi
fi

echo "Running: python Main_clean.py ${ARGS[*]}"
python Main_clean.py "${ARGS[@]}"

