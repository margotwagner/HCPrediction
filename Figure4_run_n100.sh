python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --savename Elman_SGD/Remap_predloss/N100T100/Ns100_SeqN100_predloss_full

python Main_s4.py --input data/Ns100_SeqN100/Ns100_SeqN100_2Batch.pth.tar --batch-size 2 --net ElmanRNN_tp1 --pred 1 --fixi 1 --savename Elman_SGD/Remap_predloss/N100T100/Ns100_SeqN100_2Batch_predloss

python Main_clean.py --input data/Ns100_SeqN100/Ns100_SeqN100_1.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --rnn_act relu --ac_output sigmoid --n 100 --hidden-n 100 --epoch 100000 --savename Elman_SGD/Remap_predloss/N100T100_relu_fixio/pred_relu

python Main_clean.py --input data/Ns100_SeqN100/Ns100_SeqN100_2.pth.tar --epoch 2000 -p 100 --resume Elman_SGD/Remap_predloss/N100T100_relu_fixio/pred_relu.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --n 100 --hidden-n 100 --rnn_act relu --ac_output sigmoid --savename Elman_SGD/Remap_predloss/N100T100_relu_fixio/stages/remap_s0

python Main_clean.py --input data/Ns100_SeqN100/Ns100_SeqN100_2.pth.tar --epoch 50000 -p 5000 --ae 1 --fixi 2 --fixo 2 --pred 1 --n 100 --hidden-n 100 --rnn_act relu --ac_output sigmoid --resume Elman_SGD/Remap_predloss/N100T100_relu_fixio/stages/remap_s0.pth.tar --savename Elman_SGD/Remap_predloss/N100T100_relu_fixio/stages/remap_s1

python Main_clean.py --input data/Ns100_SeqN100/Ns100_SeqN100_1_5per.pth.tar --epoch 2000 -p 100 --resume Elman_SGD/Remap_predloss/N100T100_relu_fixio/pred_relu_big.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --n 100 --hidden-n 100 --rnn_act relu --clamp_norm 0.5 --ac_output sigmoid --savename Elman_SGD/Remap_predloss/N100T100_relu_fixio/F5per_stages/remap_s0

python Main_clean.py --input data/Ns100_SeqN100/Ns100_SeqN100_1_5per.pth.tar --epoch 10000 -p 5000 --ae 1 --fixi 2 --fixo 2 --pred 1 --n 100 --hidden-n 100 --rnn_act relu --clamp_norm 0.5 --ac_output sigmoid --resume Elman_SGD/Remap_predloss/N100T100_relu_fixio/F5per_stages/remap_s0.pth.tar --savename Elman_SGD/Remap_predloss/N100T100_relu_fixio/F5per_stages/remap_s1

for noise in 10 20 30 40 50  
do  
python Main_clean.py --input data/Ns100_SeqN100/Ns100_SeqN100_1_$((noise))per.pth.tar --epoch 20000 -p 1000 --resume Elman_SGD/Remap_predloss/N100T100_relu_fixio/pred_relu_big.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --n 100 --hidden-n 100 --clamp_norm 0.5 --rnn_act relu --ac_output sigmoid --savename Elman_SGD/Remap_predloss/N100T100_relu_fixio/F$((noise))per_stages/remap_s0  
done
