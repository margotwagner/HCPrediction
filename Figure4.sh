# Good runs that only have 1 variant
python make_figures.py --just 4 \
  --conditions "./runs/ElmanRNN/random-init/random_n100" \
  --figdir ./figs/fig4 --figtag baseline --vmin -2 --vmax 2

python make_figures.py --just 4 \
  --conditions "./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro" \
  --figdir ./figs/fig4 --figtag identity --vmin -2 --vmax 2

# Best alphas
python make_figures.py --just 4 \
  --conditions "./runs/ElmanRNN/shift-variants/shift/frobenius/sym0p50/shift_n100_fro_sym0p50" \
  --figdir ./figs/fig4 --figtag shift_sym0p50 --vmin -2 --vmax 2

python make_figures.py --just 4 \
  --conditions "./runs/ElmanRNN/shift-variants/shift/frobenius/sym0p75/shift_n100_fro_sym0p75" \
  --figdir ./figs/fig4 --figtag shift_sym0p75 --vmin -2 --vmax 2

python make_figures.py --just 4 \
  --conditions "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym0p50/shiftcyc_n100_fro_sym0p50" \
  --figdir ./figs/fig4 --figtag shiftcyc_sym0p50 --vmin -2 --vmax 2

python make_figures.py --just 4 \
  --conditions "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p50/shiftcycmh_n100_fro_sym0p50" \
  --figdir ./figs/fig4 --figtag shiftcycmh_sym0p50 --vmin -2 --vmax 2

python make_figures.py --just 4 \
  --conditions "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p75/shiftcycmh_n100_fro_sym0p75" \
  --figdir ./figs/fig4 --figtag shiftcycmh_sym0p75 --vmin -2 --vmax 2

python make_figures.py --just 4 \
  --conditions "./runs/ElmanRNN/mh-variants/shifted/frobenius/sym0p50/shiftmh_n100_fro_sym0p50" \
  --figdir ./figs/fig4 --figtag shiftmh_sym0p50 --vmin -2 --vmax 2


# Multiple (untested)
python make_figures.py --just 4 \
  --cond_glob "./runs/ElmanRNN/**/sym*/identity_n100_fro*" \
              "./runs/ElmanRNN/**/sym*/shiftmh_n100_fro*" \
  --figdir ./figs/fig4 --fontsize 12


python make_figures.py --just 4 \
  --conditions "/runs/ElmanRNN/identity/frobenius/sym0p90/identity_n100_fro,\
/runs/ElmanRNN/mexhat/frobenius/sym0p70/mexhat_n100_fro" \
  --figdir ./figs/fig4
