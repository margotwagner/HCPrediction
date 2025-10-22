"""Usage:
1) Everything under a root directory:
  python aggregate_metrics.py \
      --root runs/ElmanRNN \
      --glob_offline "*_offline_metrics.csv" \
      --glob_eval "*_replay.csv,*_prediction.csv,*_openloop.csv,*_closedloop.csv" \
      --outdir runs/ElmanRNN
      
2) Single condition subtree
python aggregate_metrics.py \
  --root runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p00/shiftcycmh_n100_fro_sym0p00      
"""

python aggregate_metrics.py \
  --root runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym1p00/shiftcycmh_n100_fro_sym1p00 

python aggregate_metrics.py \
  --root runs/ElmanRNN/random-init/random_n100