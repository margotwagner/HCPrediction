nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --enforce_circulant \
  --epochs 100000 \
  --compile --amp auto --early_stop \
  --rnn_act "none" \
  --row0 ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/identity/identity_row0.npy \
  --savename ./runs/ElmanRNN/linear/identity \
  --num_runs 3 > ./logs/linear/identity.out 2>&1 &

nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --enforce_circulant \
  --epochs 100000 \
  --rnn_act "none" \
  --row0 ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/centeredmh/centeredmh_row0.npy  \
  --savename ./runs/ElmanRNN/linear/centeredmh \
  --num_runs 3 > ./logs/linear/centeredmh.out 2>&1 &

SYM="sym1p00"
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --enforce_circulant \
  --epochs 100000 \
  --rnn_act "none" \
  --row0 "./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/shift/${SYM}/shift_${SYM}_row0.npy" \
  --savename "./runs/ElmanRNN/linear/shift/${SYM}" \
  --num_runs 3 > "./logs/linear/shift_${SYM}.out" 2>&1 &

SYM="sym1p00"
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --enforce_circulant \
  --epochs 100000 \
  --rnn_act "none" \
  --row0 "./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/shiftedmh/${SYM}/shiftedmh_${SYM}_row0.npy" \
  --savename "./runs/ElmanRNN/linear/shiftedmh/${SYM}" \
  --num_runs 3 > "./logs/linear/shiftedmh_${SYM}.out" 2>&1 &