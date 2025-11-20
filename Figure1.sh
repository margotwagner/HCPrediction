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



# Pretrained
conditions=(
  "./runs/ElmanRNN/randomih/random-init/random_n100"
  "./runs/ElmanRNN/randomih/shift-variants/identity/frobenius/identity_n100_fro"
  #"./runs/ElmanRNN/randomih/learned/random_n100/frobenius/random_n100_fro_learned"
  #"./runs/ElmanRNN/randomih/learned/random_n100/raw/random_n100_learned"
  "./runs/ElmanRNN/randomih/learned/identity_n100/frobenius/identity_n100_fro_learned"
  "./runs/ElmanRNN/randomih/learned/identity_n100/raw/identity_n100_learned"
  #"./runs/ElmanRNN/randomih/learned/sorted/random_n100/frobenius/random_n100_fro_learned_sorted"
  #"./runs/ElmanRNN/randomih/learned/sorted/random_n100/raw/random_n100_learned_sorted"
  "./runs/ElmanRNN/randomih/learned/sorted/identity_n100/frobenius/identity_n100_fro_learned_sorted"
  "./runs/ElmanRNN/randomih/learned/sorted/identity_n100/raw/identity_n100_learned_sorted"
)
IFS=, conds_csv="${conditions[*]}"; unset IFS

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag raw_pretrained

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1 --figtag loglog_pretrained --fig1_logxA --fig1_logxB --fig1_logyA --fig1_logyB


# Identity weight initialization
conditions=(
  "./runs/ElmanRNN/identityih/random_baseline"
  "./runs/ElmanRNN/identityih/shift-variants/identity"
  "./runs/ElmanRNN/identityih/shift-variants/shift/sym0p50/shift_sym0p50"
  "./runs/ElmanRNN/identityih/shift-variants/cyc-shift/sym0p50/cycshift_sym0p50"
  "./runs/ElmanRNN/identityih/mh-variants/shifted/sym0p50/shiftmh_sym0p50"
)
IFS=, conds_csv="${conditions[*]}"; unset IFS
python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1/identityih --figtag logy_best --fig1_logyA --fig1_logyB

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1/identityih --figtag raw_best

# Enforcing circulant
conditions=(
  "./runs/ElmanRNN/identityih/random_baseline"
  "./runs/ElmanRNN/circulant/identity"
  "./runs/ElmanRNN/circulant/shift/sym0p50/shift_circ_sym0p50"
  "./runs/ElmanRNN/circulant/shiftedmh/sym0p50/shiftedmh_circ_sym0p50"
  "./runs/ElmanRNN/circulant/centeredmh"
)
IFS=, conds_csv="${conditions[*]}"; unset IFS
python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1/circulant --figtag logy --fig1_logyA --fig1_logyB

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1/circulant --figtag raw

# Linear activation
conditions=(
  "./runs/ElmanRNN/identityih/random_baseline"
  "./runs/ElmanRNN/circulant/identity"
  "./runs/ElmanRNN/linear/identity"
#  "./runs/ElmanRNN/circulant/shift/sym0p50"
#  "./runs/ElmanRNN/linear/shift/sym0p50"
)
IFS=, conds_csv="${conditions[*]}"; unset IFS
python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1/linear --figtag logy --fig1_logyA --fig1_logyB

python make_figures.py --just 1 \
  --conditions "$conds_csv" \
  --figdir ./figs/fig1/linear --figtag raw