from skbuild import setup
import os

print(
    "############################################################################"
)
print(os.getcwd())
setup(
    cmake_args=[
        '-DMUJOCO_ROOT_DIR:PATH=/home/curtis/.local/lib/python3.8/site-packages/mujoco'
    ],
    cmake_source_dir='plugin/',
)
