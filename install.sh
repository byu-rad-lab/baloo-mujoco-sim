#!/bin/bash
pip install -r requirements.txt
export MUJOCO_ROOT_DIR=$(pip3 show mujoco | grep Location: | cut -d ' ' -f 2)/mujoco
pip install .