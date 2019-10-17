#! /bin/bash

STEPS=1000
LR=0.002
LR_P=0.002
BATCH_SIZE=128
DATA_DIR="/home/kiran/radio_modulation/pytorch/new/pytorch_mod"


python3 run_resnet.py --classes 4 --train --file_train $DATA_DIR/qam_train4.hdf5 --steps $STEPS --batch_size $BATCH_SIZE --learning_rate $LR_P --model_path $DATA_DIR/model4.pt --filts 64,64,64,64,64,64,64,512,512,4 --results 'train_res.csv'  


python3 run_resnetp.py --classes 1 --train --file_train $DATA_DIR/qam_train4_p.hdf5  --steps $STEPS --batch_size $BATCH_SIZE --learning_rate $LR_P --model_path $DATA_DIR/model4_p.pt --filts 64,64,64,64,64,64,64,512,512,1 --results 'train_res_p.csv'


