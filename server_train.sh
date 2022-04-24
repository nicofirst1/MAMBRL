#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=mambrl
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1


#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate mambrl


# Start xvfb
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
# Export your display id
export DISPLAY=:1
export PYGLET_DEBUG_GL=True
# Check if your display settings are valid
glxinfo

echo "Start the process"
python src/trainer/FullTrainer.py