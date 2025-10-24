# Figure 1
# Single condition
python make_figures.py --just 1 \
  --conditions "./runs/ElmanRNN/identity/raw" \
  --figdir ./figs

# Multiple explicit conditions
python make_figures.py --just 1 \
  --conditions "./runs/ElmanRNN/random-init/random_n100,./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro,./runs/ElmanRNN/shift-variants/shift/frobenius/sym1p00/shift_n100_fro_sym1p00" \
  --figdir ./figs --figtag loglog --fig1_logyA --fig1_logxA --fig1_logyB --fig1_logxB

python make_figures.py --just 1 \
  --conditions "./runs/ElmanRNN/random-init/random_n100,./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro,./runs/ElmanRNN/shift-variants/shift/frobenius/sym1p00/shift_n100_fro_sym1p00" \
  --figdir ./figs --figtag log --fig1_logyA --fig1_logyB

# Glob 
python make_figures.py --just 1 \
  --cond_glob "./runs/ElmanRNN/*/*/*/*" \
  --figdir ./figs
