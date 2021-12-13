# Self-supervised learning enables excellent 3D digital subtraction angiography reconstruction from ultra-sparse 2D projection views

# Overview

The proposed self-supervised deep learning method can be used to realize excellent three-dimensional digital subtraction angiography reconstruction from a few 2D projection views for ultra-low dosage cerebrovascular imaging in clinics

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
Using a computer with a NVIDIA 2080ti GPU would takes about 15 hours for stage1.
## Dataset
The dataset involves patient privacy and ethical issues. We will release part of the dataset and pretrained model after obtaining permission.

## Training the models
for stage 1
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--stage 1 \
--train_output_path  your_train_output_path\
--train_input_path  your_train_input_path\
--result_path  your_result_path\
--view_num 6
```

for stage 2
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--stage 2 \
--train_output_path  your_train_output_path\
--train_input_path  your_train_input_path\
--result_path  your_result_path\
--last_stage_path  your_last_stage_path\
--view_num 6
```

# License
This project is open sourced under MIT license.


Please contact me (zhouzhenghong1999@gmail.com) if you have any questions!