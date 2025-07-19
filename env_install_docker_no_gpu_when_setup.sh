#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
# mamba or conda will not install cuda packages in a non-GPU environment, so we use pip to install torch with CUDA support.
set -e

echo "--- 1. Creating and activating conda environment ---"
mamba env create -f conda_environment.yaml -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate jiandecouple
pip install -y torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

echo "--- 2. Installing robosuite ---"
pip uninstall -y robosuite || true # Uninstall if exists, ignore if not
cd third_party/robosuite
pip install -e .
echo "robosuite installed from:"
python -c "import robosuite; print(robosuite.__file__)"
cd ../.. # Return to the original directory

echo "--- 3. Installing robosuite-task-zoo ---"
pip uninstall -y robosuite-task-zoo || true
cd third_party/robosuite-task-zoo
pip install -e .
echo "robosuite_task_zoo installed from:"
python -c "import robosuite_task_zoo; print(robosuite_task_zoo.__file__)"
cd ../..

echo "--- 4. Installing robomimic ---"
pip uninstall -y robomimic || true
cd third_party/robomimic
pip install -e .
echo "robomimic installed from:"
python -c "import robomimic; print(robomimic.__file__)"
cd ../..

echo "--- 5. Installing mimicgen ---"
pip uninstall -y mimicgen || true
cd third_party/mimicgen
pip install -e .
echo "mimicgen installed from:"
python -c "import mimicgen; print(mimicgen.__file__)"
cd ../..

pip install -e .

echo "--- ✅ All packages installed successfully! ---"

echo "Mujoco version:"
