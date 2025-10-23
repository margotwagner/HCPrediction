python Main_clean.py --input data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --n 100 --hidden-n 100 --ac_output sigmoid --epochs 100000 --whh_type none --num_runs 6

python Main_clean.py --input data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --n 100 --hidden-n 100 --ac_output sigmoid --epochs 100000 --whh_type cent --whh_norm frobenius --num_runs 3

python Main_clean.py \
  --input data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 2 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --ac_output sigmoid \
  --epochs 100000 --whh_type shift \
  --whh_norm frobenius --alpha 0.25 --num_runs 3

# --input: loads training tensors from here (X_mini, Target_mini)
# --ae 1: sets autoencoder mode (input=output)
# --pred 1: sets prediction mode (input=t, output=t+1)
# --fixi 2: fixes the input weight matrix at its initial random value
# --fixo 2: fixes the output weight matrix at its initial random value
# --rnn_act relu: sets the RNN activation function to ReLU (default: tanh)
# --ac_output sigmoid: sets the output activation function to sigmoid (default: softmax)


#python Main_clean.py --input data/Ns200_SeqN100_1.pth.tar --ae 1 --fixi 2 #--fixo 2 --pred 1 --rnn_act relu --ac_output sigmoid --epoch 100000 --savename #Elman_SGD/Remap_predloss/N200T100_relu_fixio/pred_relu
