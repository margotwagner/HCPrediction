# From project root (HCPrediction)
DATA=./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar
INIT_ROOT=./data/Ns100_SeqN100/hidden-weight-inits/11252025
RUN_ROOT=./runs/ElmanRNN/11252025
LOG_ROOT=./logs/phase0_circ

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

EPOCHS=100000      # or whatever you want
RUNS=3             # num_runs per condition
SEED=42            # base seed; Main_clean offsets by run_idx
CFG=cfg_tanh_linear_fi0_fo0         # tag for Phase 0 config

COMMON="--input $DATA \
  --ae 1 --pred 1 \
  --n 100 --hidden-n 100 \
  --epochs $EPOCHS \
  --rnn_act tanh \
  --act_output linear \
  --fixi 0 --fixo 0 \
  --compile --amp auto \
  --num_runs $RUNS \
  --seed $SEED \
  --print-freq 1000"

####################################
########## DENSE ###################
####################################

# Vanilla PyTorch random
nohup python Main_clean.py $COMMON \
  --enforce_circulant \
  --whh_path $INIT_ROOT/circulant/random_pytorch/seed000/row0.npy \
  --savename $RUN_ROOT/circulant/random_pytorch/$CFG \
  > $LOG_ROOT/phase0_circ_random_pytorch.out 2>&1 &


nohup python Main_clean.py $COMMON \
  --enforce_circulant \
  --whh_path $INIT_ROOT/circulant/identity/row0.npy \
  --savename $RUN_ROOT/circulant/identity/$CFG \
  > $LOG_ROOT/phase0_circ_identity.out 2>&1 &


# α = 0.0
nohup python Main_clean.py $COMMON \
  --enforce_circulant \
  --whh_path $INIT_ROOT/circulant/shift/alphasym0p00/row0.npy \
  --savename $RUN_ROOT/circulant/shift/alpha0p00/$CFG \
  > $LOG_ROOT/phase0_circ_shift_alpha0p00.out 2>&1 &

# α = 0.5
nohup python Main_clean.py $COMMON \
  --enforce_circulant \
  --whh_path $INIT_ROOT/circulant/shift/alphasym0p50/row0.npy \
  --savename $RUN_ROOT/circulant/shift/alpha0p50/$CFG \
  > $LOG_ROOT/phase0_circ_shift_alpha0p50.out 2>&1 &

# α = 1.0
nohup python Main_clean.py $COMMON \
  --enforce_circulant \
  --whh_path $INIT_ROOT/circulant/shift/alphasym1p00/row0.npy \
  --savename $RUN_ROOT/circulant/shift/alpha1p00/$CFG \
  > $LOG_ROOT/phase0_circ_shift_alpha1p00.out 2>&1 &


nohup python Main_clean.py $COMMON \
  --enforce_circulant \
  --whh_path $INIT_ROOT/circulant/mexican_hat/dog/dog2/k0/row0.npy \
  --savename $RUN_ROOT/circulant/mexican_hat/dog2/k0/$CFG \
  > $LOG_ROOT/phase0_circ_dog2_k0.out 2>&1 &


# α = 0.0
nohup python Main_clean.py $COMMON \
  --enforce_circulant \
  --whh_path $INIT_ROOT/circulant/mexican_hat/dog/dog2/k5/alphasym0p00/row0.npy \
  --savename $RUN_ROOT/circulant/mexican_hat/dog2/k5/alpha0p00/$CFG \
  > $LOG_ROOT/phase0_circ_dog2_k5_alpha0p00.out 2>&1 &

# α = 0.5
nohup python Main_clean.py $COMMON \
  --enforce_circulant \
  --whh_path $INIT_ROOT/circulant/mexican_hat/dog/dog2/k5/alphasym0p50/row0.npy \
  --savename $RUN_ROOT/circulant/mexican_hat/dog2/k5/alpha0p50/$CFG \
  > $LOG_ROOT/phase0_circ_dog2_k5_alpha0p50.out 2>&1 &

# α = 1.0
nohup python Main_clean.py $COMMON \
  --enforce_circulant \
  --whh_path $INIT_ROOT/circulant/mexican_hat/dog/dog2/k5/alphasym1p00/row0.npy \
  --savename $RUN_ROOT/circulant/mexican_hat/dog2/k5/alpha1p00/$CFG \
  > $LOG_ROOT/phase0_circ_dog2_k5_alpha1p00.out 2>&1 &

