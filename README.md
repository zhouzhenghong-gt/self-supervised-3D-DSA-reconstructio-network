# Self-supervised learning enables excellent 3D digital subtraction angiography reconstruction from ultra-sparse 2D projection views

# Overview

The proposed self-supervised deep learning method can be used to realize excellent three-dimensional digital subtraction angiography reconstruction from a few 2D projection views for ultra-low dosage cerebrovascular imaging in clinics

Note: This is a preliminary release. We will further improve our code and release some data.

The released code performs reconstruction experiments on digitally generated renderings, like DRR. Codes and methods for reconstruction of clinically acquired images is improving.


# Installation Guide
Clone the Code Repository
```
git clone https://github.com/zhouzhenghong-gt/self-supervised-3D-DSA-reconstructio-network.git
```
Install Python Denpendencies
```
cd self-supervised-3D-DSA-reconstructio-network
pip install -r requirements.txt
```

# Instructions for Use
Using a computer with a NVIDIA 3090 GPU would takes about 20 hours for stage1, and 2-3 days for stage2. Since each sample is large, CPU performance also affects speed
## Dataset
The dataset involves patient privacy and ethical issues. We will release part of the dataset and pretrained model after obtaining permission. 

## Training the models
Training model is divided into two stages, low-resolution reconstruction(128\*256\*256) and high-resolution reconstruction(395\*512\*512), where the input of high-resolution reconstruction includes low-resolution results and 2D projections

For stage 1
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--stage 1 \
--train_input_path  your_train_input_path \
--result_path  your_result_path_log_and_ckpt \
--view_num 8 \
--train_batch_size 8 \
--epoch 1500
```

Get stage1 result of training set for stage2
```
CUDA_VISIBLE_DEVICES=0 python3 make_train_predict0.py \
--stage 1 \
--val_input_path  your_train_input_path \
--predict0_path  your_result_path_of_prediction_niigz \
--module_path your_module_path \
--view_num 8 --save_niigz
```

For stage 2
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--stage 2 \
--train_input_path  your_train_input_path \
--result_path  your_result_path_of_log_and_ckpt \
--last_stage_path  your_stage1_prediction_path \
--view_num 8 \
--train_batch_size 2 \
--epoch 400 \
--leaing_rate 0.03
```

## Validating the models
Get stage1 result of validation set for stage2
```
CUDA_VISIBLE_DEVICES=0 python3 make_train_predict0.py \
--stage 1 \
--val_input_path  your_val_input_path \
--module_path your_module_path \
--predict0_path  your_result_path_of_prediction_niigz \
--view_num 8 --save_niigz
```

For stage2 validation
```
CUDA_VISIBLE_DEVICES=0 python3 validate.py \
--stage 2 \
--val_input_path  your_val_input_path \
--module_path your_module_path \
--result_path  /your_result_path_of_prediction \
--last_stage_path  your_stage1_prediction_path \
--view_num 8 
```

# Acknowledgements
Some code references [abdominal-multi-organ-segmentation](https://github.com/assassint2017/abdominal-multi-organ-segmentation)

# License
This project is open sourced under MIT license.

Please contact me (zhouzhenghong1999@gmail.com) if you have any questions!
