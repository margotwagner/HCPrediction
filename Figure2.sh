# Excluded conditions -- bad runs
bad_runs=(
)

# All conditions -- good runs
glob_conditions=(
  "./runs/ElmanRNN/shift-variants/shift/frobenius/sym*/shift_n100_fro_sym*"
  "./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym*/shiftcyc_n100_fro_sym*"
  "./runs/ElmanRNN/mh-variants/shifted/frobenius/sym*/shiftmh_n100_fro_sym*"
  "./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym*/shiftcycmh_n100_fro_sym*"
)

# join with commas into one argument
IFS=, conds_csv="${glob_conditions[*]}"; unset IFS

python make_figures.py --just 2 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig2


python make_figures.py --just 2 \
  --conditions "./runs/ElmanRNN/shift-variants/shift/frobenius/sym*/shift_n100_fro_sym*,./runs/ElmanRNN/shift-variants/shift-cyc/frobenius/sym*/shiftcyc_n100_fro_sym*,./runs/ElmanRNN/mh-variants/shifted/frobenius/sym*/shiftmh_n100_fro_sym*,./runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/sym*/shiftcycmh_n100_fro_sym*" \
  --figdir ./figs/fig2
