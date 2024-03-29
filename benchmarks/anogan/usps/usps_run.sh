#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=s1928563

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/programs/miniconda3/bin/activate anogan

for i in {0..9}
do
    echo ${i}
    python train_wgangp.py --training_label ${i} --seed 49 --n_epochs 20 --img_size 16
    python train_encoder_izif.py --training_label ${i} --seed 49 --n_epochs 20 --img_size 16
    python test_anomaly_detection.py --training_label ${i} --n_epochs 20 --img_size 16

    python train_wgangp.py --training_label ${i} --seed 49 --n_epochs 100 --img_size 16
    python train_encoder_izif.py --training_label ${i} --seed 49 --n_epochs 100 --img_size 16
    python test_anomaly_detection.py --training_label ${i} --n_epochs 100 --img_size 16

    python train_wgangp.py --training_label ${i} --seed 49 --n_epochs 150 --img_size 16
    python train_encoder_izif.py --training_label ${i} --seed 49 --n_epochs 150 --img_size 16
    python test_anomaly_detection.py --training_label ${i} --n_epochs 150 --img_size 16

    python train_wgangp.py --training_label ${i} --seed 49 --n_epochs 200 --img_size 16
    python train_encoder_izif.py --training_label ${i} --seed 49 --n_epochs 200 --img_size 16
    python test_anomaly_detection.py --training_label ${i} --n_epochs 200 --img_size 16

done