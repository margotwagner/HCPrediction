## Single run Gaussian input
# Standard He initialization
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --output_dir Elman_SGD/Remap_predloss/N100T100/he/gaussian/single-run/Ns100_SeqN100_predloss_full

# Shift
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/shift/gaussian/single-run/hidden-weights/ --savename Elman_SGD/Remap_predloss/N100T100/shift/gaussian/single-run/Ns100_SeqN100_predloss_full

# Cyclic shift
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cyclic_shift_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/cyclic-shift/gaussian/single-run/hidden-weights/ --savename Elman_SGD/Remap_predloss/N100T100/cyclic-shift/gaussian/single-run/Ns100_SeqN100_predloss_full

# Cyclic Mexican hat
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cmh_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/cmh/gaussian/single-run/hidden-weights/ --savename Elman_SGD/Remap_predloss/N100T100/cmh/gaussian/single-run/Ns100_SeqN100_predloss_full

# Multiple runs Gaussian inputs
# TODO: fix the folders once the multiruns have completed
# Standard He initialization
for i in $(printf "%02d\n" {2..9}); do nohup python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --output_dir Elman_SGD/Remap_predloss/N100T100/he/hidden-weights/multiruns/run_$i --savename Elman_SGD/Remap_predloss/N100T100/he/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Shift initialization
for i in $(printf "%02d\n" {2..9}); do nohup python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/shift/hidden-weights/multiruns/run_$i --savename Elman_SGD/Remap_predloss/N100T100/shift/multiruns/run_$i/Ns100_SeqN100_predloss_full; done

# Cyclic shift initialization
for i in $(printf "%02d\n" {2..9}); do nohup python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cyclic_shift_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/cyclic-shift/hidden-weights/multiruns/run_$i --savename Elman_SGD/Remap_predloss/N100T100/cyclic-shift/multiruns/run_$i/Ns100_SeqN100_predloss_full; done



## Single run One-hot input
# He initialization
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --output_dir Elman_SGD/Remap_predloss/N100T100/he/onehot/hidden-weights/ --savename Elman_SGD/Remap_predloss/N100T100/he/onehot/Ns100_SeqN100_predloss_full

# Shift initialization
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/shift/onehot/hidden-weights/ --savename Elman_SGD/Remap_predloss/N100T100/shift/onehot/Ns100_SeqN100_predloss_full

# Cyclic shift initialization
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1hot.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_cyclic_shift_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/cyclic-shift/onehot/hidden-weights/ --savename Elman_SGD/Remap_predloss/N100T100/cyclic-shift/onehot/Ns100_SeqN100_predloss_full