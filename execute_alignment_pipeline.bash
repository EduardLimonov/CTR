#!/bin/bash
# Скрипт запускается на узле

source .venv/bin/activate


source_patch="align_pipeline"
dataset_size=10000
dataset_path="../resource/HKR"
light_dataset_path="datasets/dataset.hdf5"
checkpoint_path="learn_output"
db_path="datasets/symb_db.hdf5"
num_workers=8

cd $source_patch
python3 create_dataset.py $dataset_path $dataset_size $light_dataset_path
python3 learn_model.py $light_dataset_path $checkpoint_path $num_workers
python3 create_symbol_db.py $light_dataset_path $checkpoint_path/model.ckpt $num_workers $db_path


cd ..
