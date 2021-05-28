#!/bin/bash
#SBATCH --job-name=ANGLEPGAN-GPU        # <image_size>,<nb_node>,<nb_MPI_task>,<nb_gpu_per_node>,<number_position>,<number_epoch
#SBATCH -N 16 
#SBATCH --ntasks-per-node=2
#SBATCH --output=output.out   # output filename
#SBATCH --error=output.err    # error filename

NETWORK_ARCH=pgan
TMPDIR='./TMPDIR'
export OMP_NUM_THREADS=34

#conda activate tf1-gpu
#source /opt/opt/intel/impi/2019.4.243/parallel_studio_xe_2019/bin/psxevars.sh
#source /opt/intel/ics/2020u1/mkl/bin/mklvars.sh intel64
#source /opt/intel/ics/2020u1/bin/compilervars.sh intel64

source /opt/intel/oneAPI/2021.1.2/mpi/latest/env/vars.sh
source /opt/intel/oneAPI/2021.1.2/mkl/latest/env/vars.sh
source /opt/crtdc/gcc/9.3.0/VARS.sh

#srun -n 1 --ntasks-per-node 1 cp classify_image_graph_def.pb $TMPDIR


#mpirun -bootstrap ssh -genv NCCL_DEBUG=INFO -genv HOROVOD_MPI_THREADS_DISABLE=1 -genv LD_LIBRARY_PATH=$LD_LIBRARY_PATH -genv TF_USE_CUDNN=0 \
mpirun --map-by ppr:1:socket:PE=36 -genv HOROVOD_MPI_THREADS_DISABLE=1 -genv LD_LIBRARY_PATH=$LD_LIBRARY_PATH  -genv OMP_NUM_THREAD=$OMP_NUM_THREADS \
python -u main.py $NETWORK_ARCH ~/scratch/CERN_anglegan/numpy/images/ \
--start_shape '(1, 2, 4, 4)' --final_shape '(1, 64, 128, 128)' \
--scratch_path ~/scratch/CERN_anglegan/scratch/home/achaibi/ --logdir ~/scratch/applications/cern/4range01 \
--horovod --num_inter_ops 1 --base_batch_size 128 --max_global_batch_size 2048 \
--starting_phase 4 --ending_phase 5 \
--latent_dim 256 --first_conv_nfilters 128 --network_size xs --starting_alpha 1 --loss_fn anglegan2 --gp_weight 10 --noise_stddev 0.01 \
--d_lr 0.005298  --d_lr_increase=linear --d_lr_decrease=exponential --d_lr_rise_niter 32768 --d_lr_decay_niter 98304 \
--g_lr 0.005408 --g_lr_increase=linear --g_lr_decrease=exponential --g_lr_rise_niter 65536 --g_lr_decay_niter 65536  \
--checkpoint_every_nsteps 50000 --summary_large_every_nsteps 128 --summary_small_every_nsteps 64 \
--data_mean 0 --data_stddev 11 \
--disable_compute_metrics_validation --disable_compute_metrics_test --calc_metrics --metrics_every_nsteps 4048 --compute_FID --metrics_batch_size=32 --num_metric_samples 32 > output.log
#--mixing_nimg 131072 --stabilizing_nimg 131072 --data_mean 0.0011767614643445087 --data_stddev 0.04581856631027229 






