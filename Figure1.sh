# Figure 1
# Single condition
python make_figures.py --just 1 \
  --conditions "./runs/ElmanRNN/identity/raw" \
  --figdir ./figs

# Excluded conditions -- bad runs
bad_runs=(
  "./runs/ElmanRNN/mh-variants/cent-cyc/frobenius/centcycmh_n100_fro"
  "./runs/ElmanRNN/mh-variants/cent/frobenius/centmh_n100_fro"
  "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p50/shiftcycmh_n100_fro_sym0p50"
)

# All conditions -- good runs
conditions=(
  "./runs/ElmanRNN/random-init/random_n100"
  "./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro"
  "./runs/ElmanRNN/shift-variants/shift/frobenius/sym0p50/shift_n100_fro_sym0p50"
  "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym0p50/shiftcyc_n100_fro_sym0p50"
  "./runs/ElmanRNN/mh-variants/shifted/frobenius/sym0p50/shiftmh_n100_fro_sym0p50"
)

# join with commas into one argument
IFS=, conds_csv="${conditions[*]}"; unset IFS

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag logx_all --fig1_logxA --fig1_logxB

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag logy_all --fig1_logyA --fig1_logyB

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag loglog_all --fig1_logxA --fig1_logxB --fig1_logyA --fig1_logyB

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag raw
