python sweep_noise_hidden.py \
    --variant identity \
    --hidden-n 100 \
    --alpha0 0.50 \
    --save-root ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/noisy/identity/frobenius \
    --base-name identity_n100_fro \
    --noise-stds 0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2 \
    --seed 42

python sweep_noise_hidden.py \
    --variant shift-cyc \
    --hidden-n 100 \
    --alpha0 0.50 \
    --save-root ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/noisy/shiftcyc/frobenius \
    --base-name shiftcyc_n100_fro \
    --noise-stds 0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2 \
    --seed 42