# 1. Evaluate
python evaluate.py \
  --base-dir ./runs/ElmanRNN/noisy/identity/frobenius/sym0p50/nstd0p001/identity_n100_fro_sym0p50_nstd0p001 \
  --runs 0-2 \
  --mode all \
  --csv ./runs/ElmanRNN/noisy/identity/frobenius/sym0p50/nstd0p001/identity_n100_fro_sym0p50_nstd0p001/identity_n100_fro_sym0p50_nstd0p001_eval.csv

python evaluate.py \
  --base-dir ./runs/ElmanRNN/noisy/identity/frobenius/sym0p50/nstd0p010/identity_n100_fro_sym0p50_nstd0p010 \
  --runs 0-2 \
  --mode all \
  --csv ./runs/ElmanRNN/noisy/identity/frobenius/sym0p50/nstd0p010/identity_n100_fro_sym0p50_nstd0p010/identity_n100_fro_sym0p50_nstd0p010_eval.csv

python evaluate.py \
  --base-dir ./runs/ElmanRNN/noisy/identity/frobenius/sym0p50/nstd0p100/identity_n100_fro_sym0p50_nstd0p100 \
  --runs 0-2 \
  --mode all \
  --csv ./runs/ElmanRNN/noisy/identity/frobenius/sym0p50/nstd0p100/identity_n100_fro_sym0p50_nstd0p100/identity_n100_fro_sym0p50_nstd0p100_eval.csv

# 2. Run offline metrics generation
python offline_metrics.py --ckpt ./runs/ElmanRNN/noisy/identity/frobenius/sym0p50/

# 3. Aggregate across conditions
python aggregate_metrics.py --root "./runs/ElmanRNN/noisy/identity/frobenius/sym0p50"