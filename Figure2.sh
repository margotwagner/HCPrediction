# Multiple Conditions
python make_figures.py --just 2 \
  --cond_glob "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym*/shiftcyc_n100_fro_sym*" \
  "./runs/ElmanRNN/shift-variants/shift/frobenius/sym*/shift_n100_fro_sym*" \
  "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym*/shiftcycmh_n100_fro_sym*" \
  "./runs/ElmanRNN/mh-variants/shifted/frobenius/sym*/shiftmh_n100_fro_sym*" \
  --figdir ./figs/fig2 --figtag all

# Single conditon
python make_figures.py --just 2 \
  --cond_glob "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym*/shiftcycmh_n100_fro_sym*" \
  --figdir ./figs/fig2 --figtag shiftcycmh

  python make_figures.py --just 2 \
  --cond_glob "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym*/shiftcyc_n100_fro_sym*" \
  --figdir ./figs/fig2 --figtag shiftcyc