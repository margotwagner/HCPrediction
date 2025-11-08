python Main_clean.py --input data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --n 100 --hidden-n 100 --act_output sigmoid --epochs 100000 --whh_type none --num_runs 6

nohup python Main_clean.py --input data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --n 100 --hidden-n 100 --act_output sigmoid --epochs 100000 --whh_type identity --whh_norm frobenius --num_runs 3 --noisy & 

python Main_clean.py \
  --input data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 2 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --epochs 100000 --whh_type shift \
  --whh_norm frobenius --alpha 1.0 --num_runs 3

# --input: loads training tensors from here (X_mini, Target_mini)
# --ae 1: sets autoencoder mode (input=output)
# --pred 1: sets prediction mode (input=t, output=t+1)
# --fixi 2: fixes the input weight matrix at its initial random value
# --fixo 2: fixes the output weight matrix at its initial random value
# --rnn_act relu: sets the RNN activation function to ReLU (default: tanh)
# --act_output sigmoid: sets the output activation function to sigmoid (default: softmax)


#python Main_clean.py --input data/Ns200_SeqN100_1.pth.tar --ae 1 --fixi 2 #--fixo 2 --pred 1 --rnn_act relu --act_output sigmoid --epoch 100000 --savename #Elman_SGD/Remap_predloss/N200T100_relu_fixio/pred_relu

# From the learned weights
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 2 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --epochs 100000 \
  --whh_path ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/learned/identity_n100/frobenius/identity_n100_fro_learned.npy \
  --num_runs 3 > identity_n100_fro_learned.out 2>&1 & 

nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 2 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --epochs 100000 \
  --whh_path ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/learned/random_n100/frobenius/random_n100_fro_learned.npy \
  --num_runs 3 > random_n100_fro_learned.out 2>&1 & 

nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 2 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --epochs 100000 \
  --whh_path ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/learned/sorted/identity_n100/raw/identity_n100_learned_sorted.npy \
  --num_runs 3 > identity_n100_learned_sorted.out 2>&1 & 

nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
    --ae 1 --fixi 2 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --epochs 100000 \
  --whh_path ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/learned/sorted/random_n100/raw/random_n100_learned_sorted.npy \
  --num_runs 3 > random_n100_learned_sorted.out 2>&1 & 

# Noisy initializations
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 2 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --epochs 100000 \
  --whh_path ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/noisy/identity/frobenius/sym0p50/nstd0p100/identity_n100_fro_sym0p50_nstd0p100.npy \
  --num_runs 3 > ./logs/identity_n100_fro_sym0p50_nstd0p100.out 2>&1 & 

# Identity encoding initializations
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --epochs 100000 \
  --whh_path ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/random/random_baseline.npy \
  --savename ./runs/ElmanRNN/identityih/random_baseline \
  --num_runs 7 > ./logs/identityih/random_baseline.out 2>&1 &

nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --epochs 100000 \
  --whh_path ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/shift-variants/identity/identity.npy \
  --savename ./runs/ElmanRNN/identityih/shift-variants/identity \
  --num_runs 7 > ./logs/identityih/identity.out 2>&1 &

nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --epochs 100000 \
  --whh_path ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/shift-variants/cyc-shift/sym1p00/cycshift_sym1p00.npy \
  --savename ./runs/ElmanRNN/identityih/shift-variants/cyc-shift/sym1p00/cycshift_sym1p00 \
  --num_runs 7 > ./logs/identityih/cycshift_sym1p00.out 2>&1 &