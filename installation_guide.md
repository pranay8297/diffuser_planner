# https://github.com/openai/mujoco-py/issues/682#issuecomment-2572292712

Installing Mujoco_Py

make sure you use Miniforge as your Conda environment # VVIMP

using miniforge3 and python 3.8.20

pip install cffi==1.14 and pip install Cython==3.0.0a10


install glfw via brew install glfw. Note the location for the installation
download MuJoCo2.1.1 image that ends with a *.dmg. The new mujoco2.1.1 is released as a Framework. You can copy the MuJoCo.app into /Applications/ folder.


## Follow These Steps as is

mkdir -p $HOME/.mujoco/mujoco210
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include

mkdir -p $HOME/.mujoco/mujoco210/bin

ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib

sudo ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib /usr/local/lib/

# For M1 (arm64) mac users:
# brew install glfw
ln -sf /opt/homebrew/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin

# remove old installation
rm -rf /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/site-packages/mujoco_py

# which python
# exit

export CC=/opt/homebrew/bin/gcc-11         # see https://github.com/openai/mujoco-py/issues/605
pip install mujoco-py && python -c 'import mujoco_py'

# Now download the source code of Mujoco-2.1.1 from releases of deepmind/mujoco
# copy the include folder from source code directory to $HOME/.mujoco/mujoco210/

After doing this, just run the below command

python -c 'import mujoco_py'

it should work fine

--------- 

After this, install each of it separately

conda install pytorch::pytorch torchvision torchaudio -c pytorch

For d4rl we need mjrl and pybullet - Visit their github, follow their installation instructions and install them before installing d4rl

Even for d4rl, we need to go to their github page and follow the installation guide

    - numpy
    - gym==0.18.0
    - mujoco-py==2.0.2.13
    - matplotlib==3.3.4
    - torch==1.9.1+cu111
    - typed-argument-parser
    - git+https://github.com/Farama-Foundation/d4rl@f2a05c0d66722499bf8031b094d9af3aea7c372b#egg=d4rl
    - scikit-image==0.17.2
    - scikit-video==1.1.11
    - gitpython
    - einops
    - ffmpeg
    - ffprobe
    - pillow
    - tqdm
    - pandas
    - wandb
    - flax >= 0.3.5
    - jax <= 0.2.21
    - ray==2.0.0
    - crcmod # for fast gsutil rsync on large files
    - google-api-python-client
    - cryptography
    - git+https://github.com/JannerM/doodad.git@janner
    - gdown>=4.6.0



____ IRRELEVENT 

Dataset download location: /Users/pranaybindela/.d4rl/datasets/halfcheetah_medium_expert-v2.hdf5

