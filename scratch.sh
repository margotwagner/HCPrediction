BASE="./runs/ElmanRNN/linear/identity"

# 1. Evaluate
python evaluate.py \
  --base-dir "${BASE}" \
  --runs 0-2 \
  --mode all \
  --csv "${BASE}/identity_eval.csv"

# 2. Run offline metrics generation
python offline_metrics.py --ckpt "${BASE}"

# 3. Aggregate across conditions
python aggregate_metrics.py --root "${BASE}"

# ----------------------------------------------------------------
# ----------------------------------------------------------------

# Set variables
BASE="./runs/ElmanRNN/circulant/shiftedmh"
SYM="sym1p00"
FULL="${BASE}/${SYM}"

# 1. EVALUATE
python evaluate.py \
  --base-dir "${FULL}" \
  --runs 0-2 \
  --mode all \
  --csv "${FULL}/${SYM}_eval.csv"

# 2. OFFLINE METRICS
python offline_metrics.py \
  --ckpt "${FULL}"

# 3. AGGREGATE
python aggregate_metrics.py \
  --root "${FULL}"
