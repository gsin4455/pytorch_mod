#! /bin/bash


DATA_DIR="/home/kiran/radio_modulation/pytorch/new/pytorch_mod"
BATCH_SIZE=128

#run inference on binary classifier
python3 run_resnet.py --file_train $DATA_DIR/qam_train4.hdf5 --batch_size $BATCH_SIZE --results "test_result.csv" --model_path $DATA_DIR/model4_resnet.pt


python3 run_resnet.py --file_train $DATA_DIR/qam_train4.hdf5 --batch_size $BATCH_SIZE  --results "test_results_p2.csv" --model_path $DATA_DIR/model4_resnet_p.pt

