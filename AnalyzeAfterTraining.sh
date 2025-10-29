"""
# 1.) TRAIN
# example: random-init, next-step prediction, 100k epochs, 3 runs
python Main_clean.py --input data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --n 100 --hidden-n 100 --pred 1 --ac_output sigmoid \
  --epochs 100000 --whh_type none --runs 3

./runs/ElmanRNN/random-init/random_n100/
  run_00/random_n100.pth.tar
  run_01/random_n100.pth.tar
  run_02/random_n100.pth.tar
  ... (logs, loss.png)
"""

# 2.) EVALUATE
# A) Evaluate all runs for the condition
# BASELINE
python evaluate.py \
  --base-dir ./runs/ElmanRNN/random-init/random_n100 \
  --runs 0-5 \
  --mode all \
  --csv ./runs/ElmanRNN/random-init/random_n100/random_n100_eval.csv

# IDENTITY
python evaluate.py \
  --base-dir ./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro \
  --runs 0-2 \
  --mode all \
  --csv ./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro/identity_n100_fro_eval.csv

# NOISY IDENTITY
python evaluate.py \
  --base-dir ./runs/ElmanRNN/noisy/shift-variants/identity/frobenius/identity_n100_fro_noisy \
  --runs 0-2 \
  --mode all \
  --csv ./runs/ElmanRNN/noisy/shift-variants/identity/frobenius/identity_n100_fro_noisy/identity_n100_fro_eval_noisy.csv

# CENTERED MH
python evaluate.py \
  --base-dir ./runs/ElmanRNN/mh-variants/cent/frobenius/centmh_n100_fro \
  --runs 0-2 \
  --mode all \
  --csv ./runs/ElmanRNN/mh-variants/cent/frobenius/centmh_n100_fro/centmh_n100_fro_eval.csv

# SHIFTED MH
python evaluate.py \
 --base-dir ./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym1p00/shiftcycmh_n100_fro_sym1p00 \
 --runs 0-2 \
 --mode all \
 --csv runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym1p00/shiftcycmh_n100_fro_sym1p00/shiftcycmh_n100_fro_sym1p00_eval.csv

# SHIFT
python evaluate.py \
 --base-dir ./runs/ElmanRNN/shift-variants/shift/frobenius/sym0p75/shift_n100_fro_sym0p75 \
 --runs 0-2 \
 --mode all \
 --csv runs/ElmanRNN/shift-variants/shift/frobenius/sym0p75/shift_n100_fro_sym0p75/shift_n100_fro_sym0p75_eval.csv

# B) Evaluate a single checkpoint
#python evaluate.py --ckpt ./runs/ElmanRNN/random-init/random_n100/run_00/#random_n100.pth.tar \
#  --mode all --csv ./runs/ElmanRNN/random-init/random_n100/run_00_eval.csv

# 3.) Run offline metrics generation
python offline_metrics.py --ckpt ./runs/ElmanRNN/random-init/random_n100/

python offline_metrics.py --ckpt ./runs/ElmanRNN/mh-variants/shifted/

python offline_metrics.py --ckpt ./runs/ElmanRNN/shift-variants/identity/

python offline_metrics.py --ckpt ./runs/ElmanRNN/noisy/shift-variants/identity/

# 4.) Aggregate across runs (condition-level tables)
python aggregate_metrics.py --root ./runs/ElmanRNN/random-init/random_n100

python aggregate_metrics.py --root ./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro

python aggregate_metrics.py --root ./runs/ElmanRNN/noisy/shift-variants/identity/frobenius/identity_n100_fro_noisy

python aggregate_metrics.py --root ./runs/ElmanRNN/mh-variants/cent-cyc/frobenius/centcycmh_n100_fro

python aggregate_metrics.py \
  --root "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym1p00/shiftcycmh_n100_fro_sym1p00" 

python aggregate_metrics.py \
  --root "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym0p75/shiftcyc_n100_fro_sym0p75" 