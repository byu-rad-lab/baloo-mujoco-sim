#!/bin/bash
pip3 install -e . #to symlink, should also install mujoco as dependency

# build plugins and install to mujoco python installation directory
message="Building and installing plugins to mujoco python installation directory"
echo $message

set PYTHON_INSTALL_LOCATION=$(pip3 show mujoco | grep Location: | cut -d ' ' -f 2)
set MUJOCO_ROOT_DIR="${PYTHON_INSTALL_LOCATION}/mujoco"

echo ${MUJOCO_ROOT_DIR}

rm -rf "./plugin/build"
mkdir "./plugin/build"
cd "./plugin/build"
cmake .. -DMUJOCO_ROOT_DIR=${MUJOCO_ROOT_DIR} -DCMAKE_BUILD_TYPE=Release
make install


# # optionally download c++ version of mujoco 
# MUJOCO_VERSION=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('mujoco'))")
# BIN_URL="https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"

# # Download the file
# curl -L -o "mujoco-${VERSION}.tar.gz" "$URL"

# # Extract the tar.gz file
# tar -xzf "mujoco-${VERSION}.tar.gz"

# # Optionally, clean up
# rm "mujoco-${VERSION}.tar.gz"


# force regeneration of xml model on install
rm -f "${PYTHON_INSTALL_LOCATION}/baloo_mujoco_sim/assets/"*.xml
