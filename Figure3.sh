python make_figures.py --just 3 \
  --conditions "./runs/ElmanRNN/random-init/random_n100" \
  --figdir ./figs/fig3 --figtag baseline --vmin -2 --vmax 2

python make_figures.py --just 3 \
  --conditions "./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro" \
  --figdir ./figs/fig3 --figtag identity --vmin -2 --vmax 2

python make_figures.py --just 3 \
  --conditions "./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro" \
  --figdir ./figs/fig3 --figtag identity --vmin -2 --vmax 2

python make_figures.py --just 3 \
  --conditions "./runs/ElmanRNN/mh-variants/cent-cyc/frobenius/centcycmh_n100_fro" \
  --figdir ./figs/fig3 --figtag centcycmh --vmin -2 --vmax 2

python make_figures.py --just 3 \
  --conditions "./runs/ElmanRNN/shift-variants/shift/frobenius/sym1p00/shift_n100_fro_sym1p00" \
  --figdir ./figs/fig3 --figtag shift_sym1p00 --vmin -2 --vmax 2

python make_figures.py --just 3 \
  --conditions "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym0p75/shiftcyc_n100_fro_sym0p75" \
  --figdir ./figs/fig3 --figtag shiftcyc_sym0p75 --vmin -2 --vmax 2

python make_figures.py --just 3 \
  --conditions "./runs/ElmanRNN/mh-variants/shifted/frobenius/sym0p50/shiftmh_n100_fro_sym0p50" \
  --figdir ./figs/fig3 --figtag shiftmh_sym0p50 --vmin -2 --vmax 2