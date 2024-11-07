#!/bin/bash

# build plugins and install to mujoco python installation directory
message="Building and installing plugins to mujoco python installation directory"
echo $message

PYTHON_INSTALL_LOCATION=$(pip3 show mujoco | grep Location: | cut -d ' ' -f 2)
MUJOCO_ROOT_DIR="${PYTHON_INSTALL_LOCATION}/mujoco"

echo ${MUJOCO_ROOT_DIR}

# optionally download c++ version of mujoco 
MUJOCO_VERSION=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('mujoco'))")

# Download the file
mkdir precompiled
cd precompiled
echo "Downloading precompiled mujoco version ${MUJOCO_VERSION}"
curl -L -o "mujoco-${MUJOCO_VERSION}.tar.gz" https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz

# Extract the tar.gz file
tar -xzf "mujoco-${MUJOCO_VERSION}.tar.gz"

# Optionally, clean up
rm "mujoco-${MUJOCO_VERSION}.tar.gz"


PRECOMPILED_MUJOCO_DIR="${PWD}/mujoco-${MUJOCO_VERSION}"
echo "Precompiled_MUJOCO_DIR: ${PRECOMPILED_MUJOCO_DIR}"

cd ..
rm -rf "./plugin/build"
mkdir "./plugin/build"
cd "./plugin/build"
cmake .. -DMUJOCO_ROOT_DIR=${MUJOCO_ROOT_DIR} -DCMAKE_BUILD_TYPE=Release -DPRECOMPILED_MUJOCO_DIR=${PRECOMPILED_MUJOCO_DIR}
make install

# force regeneration of xml model on install
cd ../../
echo "Regenerating xml model"
rm -f "./src/baloo_mujoco_sim/assets/"*.xml

#generate xml for use now?
python -c "import baloo_mujoco_sim"