python evaluate.py \
 --base-dir ./runs/ElmanRNN/random-init/random_n100 \
 --runs 0-5 \
 --mode all \
 --csv ./runs/ElmanRNN/random-init/random_n100/random_n100_all.csv

python evaluate.py \
 --base-dir ./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p50/shiftcycmh_n100_fro_sym0p50 \
 --runs 0-5 \
 --mode all \
 --csv runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p50/shiftcycmh_n100_fro_sym0p50_6.csv