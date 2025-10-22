"""Usage examples:
1) Single run
python offline_metrics.py --ckpt runs/ElmanRNN/random-init/random_n100/run_00/random_n100.pth.tar

2) All runs for a condition (directory)
python offline_metrics.py --ckpt runs/ElmanRNN/random-init/random_n100/

3) Glob across conditions
python offline_metrics.py --ckpt "runs/**/run_*/**.pth.tar"

4) Also dump S/A for init, middle, and last; include coarse pseudospectrum
python offline_metrics.py --ckpt runs/ElmanRNN/shift-variants/shiftcyc_n100_fro/ --save_SA first,middle,last --grid_ps
"""

python offline_metrics.py --ckpt runs/ElmanRNN/random-init/random_n100/