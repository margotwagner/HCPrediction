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

# Unconstrained
python make_figures.py --just 2 \
  --cond_glob "./runs/ElmanRNN/identityih/shift-variants/shift/sym*/shift_sym*" \
  "./runs/ElmanRNN/identityih/shift-variants/cyc-shift/sym*/cycshift_sym*" \
  "./runs/ElmanRNN/identityih/mh-variants/shifted/sym*/shiftmh_sym*" \
  --figdir ./figs/fig2/identityih

# Constrained
python make_figures.py --just 2 \
  --cond_glob "./runs/ElmanRNN/circulant/shift/sym*/shift_circ_sym*" \
  "./runs/ElmanRNN/circulant/shiftedmh/sym*/shiftedmh_circ_sym*" \
  --figdir ./figs/fig2/circulant