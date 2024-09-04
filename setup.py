import os
from setuptools.command.install import install as _install
import subprocess


from skbuild import setup

setup(
    cmake_args=[
        '-DMUJOCO_ROOT_DIR:PATH=/home/curtis/.local/lib/python3.8/site-packages/mujoco'
    ],
    cmake_source_dir='plugin',
)
