Scripts used to reproduce figures in:
https://www.biorxiv.org/content/10.1101/2022.05.19.492731v2

# NeuralEvidence (Fig.2)
* DataAccess.py: specs to download data from Allen Brain Observatory; a full guide is available at https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
* Main_clean.m: plot Fig.2BCD.

# Network training and analysis

* Build a python environment:  
`conda env export > environment.yml`

## Simulation of predictive network (Fig.3)

1. Train the models:  
* Non-predictive model  
`python Main_clean.py --input data/SeqN1T100.pth.tar --ae 1 -n 1 --fixo 3 --fixi 3 --rnn_act relu --ac_output sigmoid --savename Elman_SGD/Sigmoid/SeqN1T100_relu_fixio3 `  
* Predictive model  
`python Main_clean.py --input data/SeqN1T100.pth.tar --ae 1 -n 1 --fixi 3 --fixo 3 --pred 1 --rnn_act relu --ac_output sigmoid --savename Elman_SGD/Sigmoid/SeqN1T100_pred_relu_fixio3`  

2. Convert python output to *.mat  
`python py2mat_ext.py `

3. Plot the figure:  
`MATLAB Figure3.m`

## Simulation of CA1 and CA3 place cells (Fig.4)

1. Train the models: 
* Panel ABC  
&nbsp;&nbsp;&nbsp;&nbsp;`python Main_s4.py --input data/Ns200_SeqN100_1.pth.tar --batch-size 1 --net ElmanRNN_tp1 --pred 1 --fixi 1 --savename Elman_SGD/Remap_predloss/Ns200_SeqN100_predloss_full`

&nbsp;&nbsp;&nbsp;&nbsp;`python Main_s4.py --input data/Ns200_SeqN100_2Batch.pth.tar --batch-size 2 --net ElmanRNN_tp1 --pred 1 --fixi 1 --savename Elman_SGD/Remap_predloss/Ns200_SeqN100_2Batch_predloss`  

&nbsp;&nbsp;&nbsp;&nbsp;`python Main_clean.py --input data/Ns200_SeqN100_1.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --rnn_act relu --ac_output sigmoid --epoch 100000 --savename Elman_SGD/Remap_predloss/N200T100_relu_fixio/pred_relu`

* Panel DEF - Remapping  
&nbsp;&nbsp;&nbsp;&nbsp;`python Main_clean.py --input data/Ns200_SeqN100_2.pth.tar --epoch 2000 -p 100 --resume Elman_SGD/Remap_predloss/N200T100_relu_fixio/pred_relu.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --hidden-n 500 --rnn_act relu --ac_output sigmoid --savename Elman_SGD/Remap_predloss/N200T100_relu_fixio/stages/remap_s0`

&nbsp;&nbsp;&nbsp;&nbsp;`python Main_clean.py --input data/Ns200_SeqN100_2.pth.tar --epoch 50000 -p 5000 --ae 1 --fixi 2 --fixo 2 --pred 1 --hidden-n 500 --rnn_act relu --ac_output sigmoid --resume Elman_SGD/Remap_predloss/N200T100_relu_fixio/stages/remap_s0.pth.tar --savename Elman_SGD/Remap_predloss/N200T100_relu_fixio/stages/remap_s1`

&nbsp;&nbsp;&nbsp;&nbsp;`python Main_clean.py --input data/Ns200_SeqN100_1_5per.pth.tar --epoch 2000 -p 100 --resume Elman_SGD/Remap_predloss/N200T100_relu_fixio/pred_relu_big.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --hidden-n 500 --rnn_act relu --clamp_norm 0.5 --ac_output sigmoid --savename Elman_SGD/Remap_predloss/N200T100_relu_fixio/F5per_stages/remap_s0`

&nbsp;&nbsp;&nbsp;&nbsp;`python Main_clean.py --input Elman_SGD/Remap_predloss/Ns200_SeqN100_1_5per.pth.tar --epoch 10000 -p 5000 --ae 1 --fixi 2 --fixo 2 --pred 1 --hidden-n 500 --rnn_act relu --clamp_norm 0.5 --ac_output sigmoid --resume Elman_SGD/Remap_predloss/N200T100_relu_fixio/F5per_stages/remap_s0.pth.tar --savename Elman_SGD/Remap_predloss/N200T100_relu_fixio/F5per_stages/remap_s1`

&nbsp;&nbsp;&nbsp;&nbsp;`for noise in 10 20 30 40 50  
do  
python Main_clean.py --input Elman_SGD/Remap_predloss/Ns200_SeqN100_1_$((noise))per.pth.tar --epoch 20000 -p 1000 --resume Elman_SGD/Remap_predloss/N200T100_relu_fixio/pred_relu_big.pth.tar --ae 1 --fixi 2 --fixo 2 --pred 1 --hidden-n 500 --clamp_norm 0.5 --rnn_act relu --ac_output sigmoid --savename Elman_SGD/Remap_predloss/N200T100_relu_fixio/F$((noise))per_stages/remap_s0  
done`


2. Plot the figures:  
`python Figure4.py`


## Localization (Fig.5)
1. Traj_Generate.m: simulation code for straight line exploration (Main reference: https://www.pnas.org/doi/10.1073/pnas.2018422118)  

2. Localization_clean.py: generate model input from simulated trajectories.  

3. Train the model:  
* Non-predictive model  
`for i in {1..10}  
do  
python Main_s4.py --epochs 30000 --batch-size 5 --hidden-n 500 --net ElmanRNN --act sigmoid --rnn_act relu --gpu 1 --Hregularized 5.0 --clip 10 --ae 1 --input data/InputNs50_SeqN100_StraightTraj_Marcus_v2.pth.tar --savename Elman_SGD/GridInput/BatchTraining/PhysicalInput_v2/repeats/InputNs50_SeqN100_Marcus_HN500_H5.0_rep$i  
done `

* Predictive model  
`for i in {1..10}  
do  
python Main_s4.py --epochs 30000 --batch-size 5 --hidden-n 500 --net ElmanRNN_tp1 --pred 1 --act sigmoid --rnn_act relu --gpu 4 --Hregularized 5.0 --clip 10 --ae 1 --input data/InputNs50_SeqN100_StraightTraj_Marcus_v2.pth.tar --savename Elman_SGD/GridInput/BatchTraining/PhysicalInput_v2/repeats/InputNs50_SeqN100_Marcus_HN500_H5.0_tp1_rep$i
done`

4. Plot the figure:
`python Figure5.py`

## MNIST sequence learning (Fig.6)

1. Generate MNIST inputs:  
`python Figure6_InputPrep.py`

2. Train models:  
`python Main.py  -n 68 --input MNIST_68PC_SeqN100_Ns5.pth.tar --lr 0.0002 --pred 1 --partial 0.2 --ac_output tanh --Hregularized 1 --epochs 10000 -p 100 --savename Elman_SGD/predloss/MNIST_68PC_SeqN100_Ns5_partial`

3. Plot the figure:  
`python Figure6.py`

## Local RNN training (Fig.7)

1. Train models:  
`python Main_local.py --n 68 --hidden_n 100 --inputfile 'data/data_pca.pkl' --savename 'mnist_local' --learning_alg 'local' --epochs 500001 --lr 0.001 --t 10`

2. Plot the figure:  
`python Figure7.py`