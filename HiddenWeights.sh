# Standard He initialization
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --output_dir Elman_SGD/Remap_predloss/N100T100/he/hidden-weights --savename Elman_SGD/Remap_predloss/N100T100/he/Ns100_SeqN100_predloss_full

python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_2Batch.pth.tar --batch-size 2 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --output_dir Elman_SGD/Remap_predloss/N100T100/he/hidden-weights --savename Elman_SGD/Remap_predloss/N100T100/he/Ns100_SeqN100_2Batch_predloss

# TODO: fix Main_clean.py and run associated scripts

# Shift initialization
python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/shift/hidden-weights --savename Elman_SGD/Remap_predloss/N100T100/shift/Ns100_SeqN100_predloss_full

python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_2Batch.pth.tar --batch-size 2 --net ElmanRNN_tp1 --pred 1 --fixi 1 --hidden-n 100 --hidden_init data/Ns100_SeqN100/hidden-weight-inits/hidden_shift_n100_xavier.npy --output_dir Elman_SGD/Remap_predloss/N100T100/shift/hidden-weights --savename Elman_SGD/Remap_predloss/N100T100/shift/Ns100_SeqN100_2Batch_predloss