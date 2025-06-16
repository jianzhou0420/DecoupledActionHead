# Template
A="stack_d1",
B= "coffee_d2",
C= "three_piece_assembly_d2",
D= "stack_three_d1",
E= "square_d2",
F= "threading_d2",
G= "hammer_cleanup_d1",
H= "mug_cleanup_d1",
I= "kitchen_d1",
J= "nut_assembly_d0",
K= "pick_place_d0",
L= "coffee_preparation_d1"

#obs
# python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/${dataset}/${dataset}.hdf5 --output data/robomimic/datasets/${dataset}/${dataset}_voxel.hdf5 --num_workers=18

#action
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${dataset}/${dataset}.hdf5 -o data/robomimic/datasets/${dataset}/${dataset}_abs.hdf5 -n 18
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${dataset}/${dataset}_voxel.hdf5 -o data/robomimic/datasets/${dataset}/${dataset}_voxel_abs.hdf5 -n 18

# train

python trainer_pl_normal.py --config-name=train_diffusion_unet task_name=stack_d1 n_demo=1000
python trainer_pl_stage1.py --config-name=DP_DecoupleActionHead_stage1 n_demo=1000 dataset_path=./data/robomimic/datasets/ABC/stack_d1_coffee_d2_three_piece_assembly_d2_abs_JP2eePose.hdf5
python trainer_pl_stage2.py --config-name=DP_DecoupleActionHead_stage2 n_demo=1000 task_name=stack_d1

# debug args
logging.mode=offline task.env_runner.n_train=1 task.env_runner.n_test=1 task.env_runner.n_envs=2

python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/coffee_d2/coffee_d2.hdf5 -o data/robomimic/datasets/coffee_d2/coffee_d2_abs.hdf5 -n 12

不要用train_diffusion_unet, 这只是一个模板
用train_

#HPC

rsync -avP /data/
scp -r outputs/2025.06.10/19.12.37_DecoupleActionHead_stage1_None_1000

a1946536@p2-log-1.hpc.adelaide.edu.au:/hpcfs/users/a1946536/git_all/first/data/robomimic/datasets
