nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --enforce_circulant \
  --epochs 100000 \
  --compile --amp auto --early_stop \
  --row0 ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/identity/identity_row0.npy \
  --savename ./runs/ElmanRNN/circulant/test_es \
  --num_runs 1 > ./logs/circulant/test_es.out 2>&1 &

nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --enforce_circulant \
  --epochs 100000 \
  --row0 ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/centeredmh/centeredmh_row0.npy  \
  --savename ./runs/ElmanRNN/circulant/centeredmh \
  --num_runs 3 > ./logs/circulant/centeredmh.out 2>&1 &

SYM="sym0p75"
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --enforce_circulant \
  --epochs 100000 \
  --row0 "./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/shift/${SYM}/shift_${SYM}_row0.npy" \
  --savename "./runs/ElmanRNN/circulant/shift/${SYM}" \
  --num_runs 3 > "./logs/circulant/shift_${SYM}.out" 2>&1 &

SYM="sym0p75"
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output sigmoid \
  --enforce_circulant \
  --epochs 100000 \
  --row0 "./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/shiftedmh/${SYM}/shiftedmh_${SYM}_row0.npy" \
  --savename "./runs/ElmanRNN/circulant/shiftedmh/${SYM}" \
  --num_runs 3 > "./logs/circulant/shiftedmh_${SYM}.out" 2>&1 &

# FIXI 4 + LINEAR ACT_OUTPUT 
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output linear \
  --enforce_circulant \
  --epochs 100000 \
  --compile --amp auto \
  --row0 ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/identity/identity_row0.npy \
  --savename ./runs/ElmanRNN/id_in_linear_out \
  --num_runs 3 > ./logs/id_in_linear_out/identity.out 2>&1 &

nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output linear \
  --enforce_circulant \
  --epochs 100000 \
  --row0 ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/centeredmh/centeredmh_row0.npy  \
  --savename ./runs/ElmanRNN/id_in_linear_out/centeredmh \
  --num_runs 3 > ./logs/id_in_linear_out/centeredmh.out 2>&1 &

SYM="sym1p00"
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output linear \
  --enforce_circulant \
  --epochs 100000 \
  --row0 "./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/shift/${SYM}/shift_${SYM}_row0.npy" \
  --savename "./runs/ElmanRNN/id_in_linear_out/shift/${SYM}" \
  --num_runs 3 > "./logs/id_in_linear_out/shift_${SYM}.out" 2>&1 &

SYM="sym1p00"
nohup python Main_clean.py \
  --input ./data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar \
  --ae 1 --fixi 4 --fixo 2 --pred 1 \
  --n 100 --hidden-n 100 --act_output linear \
  --enforce_circulant \
  --epochs 100000 \
  --row0 "./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/circulant/shiftedmh/${SYM}/shiftedmh_${SYM}_row0.npy" \
  --savename "./runs/ElmanRNN/id_in_linear_out/shiftedmh/${SYM}" \
  --num_runs 3 > "./logs/id_in_linear_out/shiftedmh_${SYM}.out" 2>&1 &