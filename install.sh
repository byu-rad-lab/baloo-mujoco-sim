#!/bin/bash
pip install -r requirements.txt
export PYTHON_INSTALL_LOCATION=$(pip3 show mujoco | grep Location: | cut -d ' ' -f 2)
export MUJOCO_ROOT_DIR="${PYTHON_INSTALL_LOCATION}/mujoco"
pip install .

# force regeneration of xml model
rm -f "${PYTHON_INSTALL_LOCATION}/baloo_mujoco_sim/assets/"*.xml
