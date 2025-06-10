# Template
dataset="coffee_d2"

#obs
# python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/${dataset}/${dataset}.hdf5 --output data/robomimic/datasets/${dataset}/${dataset}_voxel.hdf5 --num_workers=18

#action
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${dataset}/${dataset}.hdf5 -o data/robomimic/datasets/${dataset}/${dataset}_abs.hdf5 -n 18
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${dataset}/${dataset}_voxel.hdf5 -o data/robomimic/datasets/${dataset}/${dataset}_voxel_abs.hdf5 -n 18

# train

python trainer_pl_normal.py --config-name=train_diffusion_unet task_name=stack_d1 n_demo=1000
python trainer_pl_stage1.py --config-name=DP_DecoupleActhonHead_stage1 n_demo=1000 dataset_path=/media/jian/ssd4t/DP/first/data/robomimic/datasets/ABC/stack_d1_coffee_d2_three_piece_assembly_d2_abs_JP2eePose.hdf5
# debug args
logging.mode=offline task.env_runner.n_train=1 task.env_runner.n_test=1 task.env_runner.n_envs=2

python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/coffee_d2/coffee_d2.hdf5 -o data/robomimic/datasets/coffee_d2/coffee_d2_abs.hdf5 -n 12

不要用train_diffusion_unet, 这只是一个模板
用train_

#HPC

rsync -avP /data/
