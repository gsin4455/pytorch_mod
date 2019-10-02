#! /bin/bash

GPUS=0,1
DATA_DIR="/home/kiran/radio_modulation/pytorch/new/pytorch_mod"
BATCH_SIZE=1024
'''
#run inference on binary classifier
python3 run_net_pkl.py --file_test $DATA_DIR/qam_data_test.pkl --batch_size $BATCH_SIZE --results "test_result.csv" --model_path $DATA_DIR/model.pt --gpus $GPUS
'''
#run inference with n nary classifier
python3 run_net_pkl.py --file_test $DATA_DIR/qam_data_p_test.pkl --batch_size $BATCH_SIZE --results "test_result_p.csv" --model_path $DATA_DIR/checkpoint.pt --gpus $GPUS

