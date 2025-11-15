sudo apt-get -y install git-lfs

git lfs install

git clone https://huggingface.co/datasets/JianZhou0420/DecoupledActionHead/ data/robomimic/datasets_abs

python scripts/convertor.py -t ABCDEFGH -c traj_eePose JP2eePose