################################################################################################################################################################################################################################################

## Single run Gaussian input
# Standard He initialization (Gaussian activation)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--output_dir Elman_SGD/Remap_predloss/N100T100/he/gaussian/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/he/gaussian/single-run/Ns100_SeqN100_predloss_full

# Shift (Gaussian)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/shift/gaussian/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/shift/gaussian/single-run/Ns100_SeqN100_predloss_full

# Cyclic shift (Gaussian)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cyclic_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cyclic-shift/gaussian/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/cyclic-shift/gaussian/single-run/Ns100_SeqN100_predloss_full

# Cyclic Mexican hat (Gaussian)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cmh_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/cmh/gaussian/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/cmh/gaussian/single-run/Ns100_SeqN100_predloss_full

# Standard Mexican hat (Gaussian)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_mh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/mh/gaussian/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/mh/gaussian/single-run/Ns100_SeqN100_predloss_full

# Cyclic tridiagonal (Gaussian)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_ctridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/ctridiag/gaussian/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/ctridiag/gaussian/single-run/Ns100_SeqN100_predloss_full

# Tridiagonal (Gaussian)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_tridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/tridiag/gaussian/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/tridiag/gaussian/single-run/Ns100_SeqN100_predloss_full

# Orthogonal (Gaussian)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_orthog_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/orthog/gaussian/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/orthog/gaussian/single-run/Ns100_SeqN100_predloss_full

################################################################################################################################################################################################################################################

## Single run One-hot input
# He initialization (one-hot activation)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar  \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--output_dir Elman_SGD/Remap_predloss/N100T100/he/onehot/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/he/onehot/single-run/Ns100_SeqN100_predloss_full

# Shift initialization (one-hot)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/shift/onehot/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/shift/onehot/single-run/Ns100_SeqN100_predloss_full

# Cyclic shift initialization (one-hot)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cyclic_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cyclic-shift/onehot/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/cyclic-shift/onehot/single-run/Ns100_SeqN100_predloss_full

# Cyclic Mexican hat (one-hot)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cmh_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/cmh/onehot/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/cmh/onehot/single-run/Ns100_SeqN100_predloss_full

# Standard Mexican hat (one-hot)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_mh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/mh/onehot/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/mh/onehot/single-run/Ns100_SeqN100_predloss_full

# Cyclic tridiagonal (one-hot)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_ctridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/ctridiag/onehot/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/ctridiag/onehot/single-run/Ns100_SeqN100_predloss_full

# Tridiagonal (one-hot)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_tridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/tridiag/onehot/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/tridiag/onehot/single-run/Ns100_SeqN100_predloss_full

# Orthogonal (one-hot)
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar --batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_orthog_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/orthog/onehot/single-run/hidden-weights/ \
--savename Elman_SGD/Remap_predloss/N100T100/orthog/onehot/single-run/Ns100_SeqN100_predloss_full

################################################################################################################################################################################################################################################

# Multiple runs Gaussian inputs
# Standard He initialization (Gaussian, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--output_dir Elman_SGD/Remap_predloss/N100T100/he/gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/he/gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Shift initialization (Gaussian, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/shift/gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/shift/gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic shift initialization (Gaussian, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cyclic_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cyclic-shift/gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/cyclic-shift/gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic Mexican Hat (Gaussian, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cmh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cmh/gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/cmh/gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Standard Mexican Hat (Gaussian, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_mh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/mh/gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/mh/gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic Tridiagonal (Gaussian, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_ctridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/ctridiag/gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/ctridiag/gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Tridiagonal (Gaussian, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_tridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/tridiag/gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/tridiag/gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Orthogonal (Gaussian, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_orthog_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/orthog/gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/orthog/gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

################################################################################################################################################################################################################################################

# Multiple runs One-hot inputs
# Standard He initialization (One-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--output_dir Elman_SGD/Remap_predloss/N100T100/he/onehot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/he/onehot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Shift initialization (one-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/shift/onehot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/shift/onehot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic shift initialization (one-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cyclic_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cyclic-shift/onehot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/cyclic-shift/onehot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic Mexican Hat (one-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cmh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cmh/onehot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/cmh/onehot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Standard Mexican Hat (one-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_mh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/mh/onehot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/mh/onehot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic Tridiagonal (one-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_ctridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/ctridiag/onehot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/ctridiag/onehot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Tridiagonal (one-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_tridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/tridiag/onehot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/tridiag/onehot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Orthogonal (one-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_orthog_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/orthog/onehot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/orthog/onehot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

################################################################################################################################################################################################################################################

# Multiple runs k-hot inputs
# Standard He initialization (k-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--output_dir Elman_SGD/Remap_predloss/N100T100/he/khot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/he/khot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Shift initialization (k-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/shift/khot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/shift/khot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic shift initialization (k-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cyclic_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cyclic-shift/khot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/cyclic-shift/khot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic Mexican Hat (k-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cmh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cmh/khot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/cmh/khot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Standard Mexican Hat (k-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_mh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/mh/khot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/mh/khot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic Tridiagonal (k-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_ctridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/ctridiag/khot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/ctridiag/khot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Tridiagonal (k-hot, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_tridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/tridiag/khot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/tridiag/khot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Orthogonal (k-hot, multiruns)
for i in $(printf "%02d\n" {5..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5hot.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_orthog_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/orthog/khot/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/orthog/khot/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

################################################################################################################################################################################################################################################

# Multiple runs small gauss (k=5) inputs
# Standard He initialization (5gauss, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5gauss.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--output_dir Elman_SGD/Remap_predloss/N100T100/he/small-gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/he/small-gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Shift initialization (5gauss, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5gauss.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/shift/small-gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/shift/small-gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic shift initialization (5gauss, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5gauss.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cyclic_shift_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cyclic-shift/small-gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/cyclic-shift/small-gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic Mexican Hat (5gauss, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5gauss.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cmh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/cmh/small-gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/cmh/small-gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Standard Mexican Hat (5gauss, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5gauss.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_mh_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/mh/small-gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/mh/small-gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic Tridiagonal (5gauss, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5gauss.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_ctridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/ctridiag/small-gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/ctridiag/small-gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Tridiagonal (5gauss, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5gauss.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_tridiag_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/tridiag/small-gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/tridiag/small-gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Orthogonal (5gauss, multiruns)
for i in $(printf "%02d\n" {0..9}); do nohup python Main_s4.py \
--input data/Ns100_SeqN100/Ns100_SeqN100_5gauss.pth.tar \
--batch-size 1 --net ElmanRNN_pytorch_module_v2 \
--pred 1 --fixi 1 --hidden-n 100 \
--seed "$i" \
--hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_orthog_n100_xavier.npy \
--output_dir Elman_SGD/Remap_predloss/N100T100/orthog/small-gaussian/multiruns/run_$i/hidden-weights \
--savename Elman_SGD/Remap_predloss/N100T100/orthog/small-gaussian/multiruns/run_$i/Ns100_SeqN100_predloss_full; done