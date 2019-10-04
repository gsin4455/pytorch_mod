#! /bin/bash

GPUS=0,1
STEPS=500
LR=0.002
LR_P=0.002
BATCH_SIZE=128
DATA_DIR="/home/kiran/radio_modulation/pytorch/new/pytorch_mod"


python3 run_resnet.py --classes 2 --train --file_train $DATA_DIR/qam_train.hdf5 --steps $STEPS --batch_size $BATCH_SIZE --learning_rate $LR_P --model_path $DATA_DIR/model.pt --gpus $GPUS --filts 64,64,64,64,64,64,64,512,512,2 --results 'train_res.csv'  


python3 run_resnet.py --classes 8 --train --file_train $DATA_DIR/qam_train_p.hdf5  --steps $STEPS --batch_size $BATCH_SIZE --learning_rate $LR_P --model_path $DATA_DIR/model_p.pt --gpus $GPUS --filts 64,64,64,64,64,64,64,512,512,8 --results 'train_res_p.csv'


