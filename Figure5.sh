python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/random-init/random_n100" \
  --fig5_time all \
  --figdir ./figs/fig5 --fontsize 12

python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro" \
  --fig5_time all \
  --figdir ./figs/fig5 --fontsize 12

# Best alphas
python make_figures.py --just 5 \
  --conditions "./runs/ElmanRNN/shift-variants/shift/frobenius/sym0p50/shift_n100_fro_sym0p50" \
  --fig5_time all \
  --figdir ./figs/fig5

python make_figures.py --just 5 \
  --conditions "./runs/ElmanRNN/shift-variants/shift/frobenius/sym0p75/shift_n100_fro_sym0p75" \
  --fig5_time all \
  --figdir ./figs/fig5

python make_figures.py --just 5 \
  --conditions "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym0p50/shiftcyc_n100_fro_sym0p50" \
  --fig5_time all \
  --figdir ./figs/fig5

python make_figures.py --just 5 \
  --conditions "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p50/shiftcycmh_n100_fro_sym0p50" \
  --fig5_time all \
  --figdir ./figs/fig5

python make_figures.py --just 5 \
  --conditions "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p75/shiftcycmh_n100_fro_sym0p75" \
  --fig5_time all \
  --figdir ./figs/fig5

python make_figures.py --just 5 \
  --conditions "./runs/ElmanRNN/mh-variants/shifted/frobenius/sym0p50/shiftmh_n100_fro_sym0p50" \
  --fig5_time all \
  --figdir ./figs/fig5


  # Multiple conditions (explicit list)
  python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/clean/.../identity_n100_fro, ./runs/ElmanRNN/clean/.../shiftcyc_n100_fro" \
  --fig5_time "first,last" \
  --figdir ./figs --figtag compare


# Multiple conditions (globs)
python make_figures.py \
  --just 5 \
  --cond_glob "./runs/ElmanRNN/clean/**/sym*/identity_n100_fro" "./runs/ElmanRNN/clean/**/sym*/shiftcyc_n100_fro" \
  --fig5_time all \
  --figdir ./figs --figtag sweep

# Unconstrained
python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/identityih/random_baseline" \
  --fig5_time all \
  --figdir ./figs/fig5/identityih --fontsize 12

python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/identityih/shift-variants/identity" \
  --fig5_time all \
  --figdir ./figs/fig5/identityih --fontsize 12

python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/identityih/shift-variants/shift/sym0p00/shift_sym0p00" \
  --fig5_time all \
  --figdir ./figs/fig5/identityih --fontsize 12

python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/identityih/shift-variants/cyc-shift/sym0p75/cycshift_sym0p75" \
  --fig5_time all \
  --figdir ./figs/fig5/identityih --fontsize 12

python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/identityih/mh-variants/shifted/sym0p00/shiftmh_sym0p00" \
  --fig5_time all \
  --figdir ./figs/fig5/identityih --fontsize 12

# Constrained
python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/circulant/identity" \
  --fig5_time all \
  --figdir ./figs/fig5/circulant --fontsize 12

python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/circulant/centeredmh" \
  --fig5_time all \
  --figdir ./figs/fig5/circulant --fontsize 12

python make_figures.py \
  --just 5 \
  --conditions "./runs/ElmanRNN/circulant/shiftedmh/sym0p75/shiftedmh_circ_sym0p75" \
  --fig5_time all \
  --figdir ./figs/fig5/circulant --fontsize 12