# From project root (HCPrediction)
DATA=./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar
INIT_ROOT=./data/Ns100_SeqN100/hidden-weight-inits/11252025
RUN_ROOT=./runs/ElmanRNN/cfg_tanh_linear_fi4_fo4_12022025
LOG_ROOT=./logs/cfg_tanh_linear_fi4_fo4_12022025/dense

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

EPOCHS=100000      # or whatever you want
RUNS=3             # num_runs per condition
SEED=42            # base seed; Main_clean offsets by run_idx
CFG=cfg_tanh_linear_fi4_fo4         # tag for Phase 0 config

COMMON="--input $DATA \
  --ae 1 --pred 1 \
  --n 100 --hidden-n 100 \
  --epochs $EPOCHS \
  --rnn_act tanh \
  --act_output linear \
  --fixi 4 --fixo 4 \
  --amp off \
  --num_runs $RUNS \
  --seed $SEED \
  --print-freq 1000"

####################################
########## DENSE ###################
####################################

# Vanilla PyTorch random
nohup python Main_clean.py $COMMON \
  --whh_path $INIT_ROOT/dense/random_pytorch/seed000/Whh.npy \
  --savename $RUN_ROOT/dense/random_pytorch/$CFG \
  > $LOG_ROOT/phase0_dense_random_pytorch.out 2>&1 &

nohup python Main_clean.py $COMMON \
  --whh_path $INIT_ROOT/dense/identity/Whh.npy \
  --savename $RUN_ROOT/dense/identity/$CFG \
  > $LOG_ROOT/phase0_dense_identity.out 2>&1 &

# α = 0.0
nohup python Main_clean.py $COMMON \
  --whh_path $INIT_ROOT/dense/shift/cyclic/alphasym0p00/Whh.npy \
  --savename $RUN_ROOT/dense/shift/cyclic/alpha0p00/$CFG \
  > $LOG_ROOT/phase0_dense_shift_alpha0p00.out 2>&1 &

# α = 0.5
nohup python Main_clean.py $COMMON \
  --whh_path $INIT_ROOT/dense/shift/cyclic/alphasym0p50/Whh.npy \
  --savename $RUN_ROOT/dense/shift/cyclic/alpha0p50/$CFG \
  > $LOG_ROOT/phase0_dense_shift_alpha0p50.out 2>&1 &

# α = 1.0
nohup python Main_clean.py $COMMON \
  --whh_path $INIT_ROOT/dense/shift/cyclic/alphasym1p00/Whh.npy \
  --savename $RUN_ROOT/dense/shift/cyclic/alpha1p00/$CFG \
  > $LOG_ROOT/phase0_dense_shift_alpha1p00.out 2>&1 &

nohup python Main_clean.py $COMMON \
  --whh_path $INIT_ROOT/dense/mexican_hat/dog/dog2/k0/Whh.npy \
  --savename $RUN_ROOT/dense/mexican_hat/dog2/k0/$CFG \
  > $LOG_ROOT/phase0_dense_dog2_k0.out 2>&1 &

# α = 0.0
nohup python Main_clean.py $COMMON \
  --whh_path $INIT_ROOT/dense/mexican_hat/dog/dog2/k5/alphasym0p00/Whh.npy \
  --savename $RUN_ROOT/dense/mexican_hat/dog2/k5/alpha0p00/$CFG \
  > $LOG_ROOT/phase0_dense_dog2_k5_alpha0p00.out 2>&1 &

# α = 0.5
nohup python Main_clean.py $COMMON \
  --whh_path $INIT_ROOT/dense/mexican_hat/dog/dog2/k5/alphasym0p50/Whh.npy \
  --savename $RUN_ROOT/dense/mexican_hat/dog2/k5/alpha0p50/$CFG \
  > $LOG_ROOT/phase0_dense_dog2_k5_alpha0p50.out 2>&1 &

# α = 1.0
nohup python Main_clean.py $COMMON \
  --whh_path $INIT_ROOT/dense/mexican_hat/dog/dog2/k5/alphasym1p00/Whh.npy \
  --savename $RUN_ROOT/dense/mexican_hat/dog2/k5/alpha1p00/$CFG \
  > $LOG_ROOT/phase0_dense_dog2_k5_alpha1p00.out 2>&1 &
