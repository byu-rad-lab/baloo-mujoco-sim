from skbuild import setup
import os
import importlib.metadata
from pathlib import Path as path
import sys

# can't get path of mujoco package because pip puts stuff in weird /tmp folder where mucodo doesn't exist.

#works, but requires path setting done in install.sh
mujoco_path = os.environ.get('MUJOCO_ROOT_DIR')

setup(
    cmake_args=[f'-DMUJOCO_ROOT_DIR:PATH={mujoco_path}'],
    cmake_source_dir='plugin/',
)
