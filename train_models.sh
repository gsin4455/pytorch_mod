#! /bin/bash

STEPS = 1000
LR = 0.005
BATCH_SIZE = 128

DATA_DIR= /home/kiran/radio_modulation/pytorch/new/pytorch_mod

#run binary classifier
python3 run_net_pkl.py --train --path $DATA_DIR --steps $STEPS --learning_rate $LR --no_filts 64,64,64,64,64,64,64,512,512,2 


#run classifier with n labels
python3 run_net_pkl.py --train --path $DATA_DIR --steps $STEPS --learning_rate $LR --no_filts 64,64,64,64,64,64,64,512,512,128

