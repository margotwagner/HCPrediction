# Figure 1
# Single condition
python make_figures.py --just 1 \
  --conditions "./runs/ElmanRNN/identity/raw" \
  --figdir ./figs

# Excluded conditions -- bad runs
bad_runs=(
  "./runs/ElmanRNN/mh-variants/cent-cyc/frobenius/centcycmh_n100_fro"
  "./runs/ElmanRNN/mh-variants/cent/frobenius/centmh_n100_fro"
  "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym0p25/shiftcycmh_n100_fro_sym0p25"
)

# All conditions -- good runs
conditions=(
  "./runs/ElmanRNN/random-init/random_n100"
  "./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro"
  "./runs/ElmanRNN/shift-variants/shift/frobenius/sym1p00/shift_n100_fro_sym1p00"
  "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym0p75/shiftcyc_n100_fro_sym0p75"
  "./runs/ElmanRNN/mh-variants/shifted/frobenius/sym0p50/shiftmh_n100_fro_sym0p50"
)

# join with commas into one argument
IFS=, conds_csv="${conditions[*]}"; unset IFS

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag logx_best --fig1_logxA --fig1_logxB

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag logy_best --fig1_logyA --fig1_logyB

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag loglog_best --fig1_logxA --fig1_logxB --fig1_logyA --fig1_logyB

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag raw_best

# NOISE
conditions=(
  "./runs/ElmanRNN/random-init/random_n100"
  "./runs/ElmanRNN/shift-variants/identity/frobenius/identity_n100_fro"
  "./runs/ElmanRNN/noisy/shift-variants/identity/frobenius/identity_n100_fro_noisy"
)
IFS=, conds_csv="${conditions[*]}"; unset IFS
python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag raw_id_noisy
python make_figures.py --just 1 \
--conditions "$conds_csv" \
--figdir ./figs/fig1 --figtag loglog_id_noisy --fig1_logxA --fig1_logxB --fig1_logyA --fig1_logyB