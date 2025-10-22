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
python evaluate.py \
  --base-dir ./runs/ElmanRNN/random-init/random_n100 \
  --runs 0-5 \
  --mode all \
  --csv ./runs/ElmanRNN/random-init/random_n100/random_n100_eval.csv

python evaluate.py \
 --base-dir ./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym1p00/shiftcycmh_n100_fro_sym1p00 \
 --runs 0-2 \
 --mode all \
 --csv runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym1p00/shiftcycmh_n100_fro_sym0p00/shiftcycmh_n100_fro_sym1p00_eval.csv

# B) Evaluate a single checkpoint
#python evaluate.py --ckpt ./runs/ElmanRNN/random-init/random_n100/run_00/#random_n100.pth.tar \
#  --mode all --csv ./runs/ElmanRNN/random-init/random_n100/run_00_eval.csv

# 3.) Run offline metrics generation
python offline_metrics.py --ckpt ./runs/ElmanRNN/random-init/random_n100/

python offline_metrics.py --ckpt ./runs/ElmanRNN/mh-variants/shifted-cyc/

# 4.) Aggregate across runs (condition-level tables)
python aggregate_metrics.py \
  --root ./runs/ElmanRNN/random-init/random_n100 \
  --out ./runs/ElmanRNN/random-init/random_n100/

# 5.) Plot results (example script)
python make_figures.py \
  --conditions "/runs/ElmanRNN/random-init/random_n100" \
  --cond_glob "/runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/*/shiftmh_n100_fro" \
  --figdir ./figs_compare_shiftedcyc_vs_random \
  --fontsize 12

# 3 explicit
  python make_figures.py \
  --conditions "/runs/ElmanRNN/random-init/random_n100" \
  --figdir ./figs_compare --fontsize 12


# 6.) Supplementary analyses (example script)
python make_figures_supp.py \
  --conditions \
    ./runs/ElmanRNN/random-init/random_n100/aggregated.csv \
    ./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p50/aggregated.csv \
  --outdir ./figures/supp_compare_random_vs_shiftedcyc_sym050 \
  --fontsize 10

  