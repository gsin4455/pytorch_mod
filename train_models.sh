#! /bin/bash
GPUS=0,1
STEPS=500
LR=0.01
LR_P=0.2
BATCH_SIZE=128

DATA_DIR="/home/kiran/radio_modulation/pytorch/new/pytorch_mod"


#run binary classifier
python3 run_net_pkl.py --train --file_train $DATA_DIR/qam_data.pkl --file_test $DATA_DIR/qam_data_test.pkl --steps $STEPS --batch_size $BATCH_SIZE --learning_rate $LR --results "train_results.csv" --filts 64,64,64,64,64,64,64,512,512,2 --model_path $DATA_DIR/model.pt --gpus $GPUS

#run classifier with n labels
python3 run_net_pkl.py --train --file_train $DATA_DIR/qam_data_p.pkl --file_test $DATA_DIR/qam_data_test.pkl --steps $STEPS --batch_size $BATCH_SIZE --learning_rate $LR_P --results "train_results_p.csv" --filts 64,64,64,64,64,64,64,512,512,128 --model_path $DATA_DIR/model_p.pt --gpus $GPUS

