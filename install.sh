#!/bin/bash

#Download correct release of ruckig
message="Installing ruckig v0.14.0 as plugin dependency..."
echo -e $message

# download ruckig
rm -rf ./plugin/ruckig_actuator/dependencies
mkdir -p ./plugin/ruckig_actuator/dependencies
cd ./plugin/ruckig_actuator/dependencies
curl -L -o "ruckig-0.14.0.tar.gz" https://github.com/pantor/ruckig/archive/refs/tags/v0.14.0.tar.gz

#extract ruckig
tar -xzf "ruckig-0.14.0.tar.gz"
rm "ruckig-0.14.0.tar.gz"

# build ruckig
cd ruckig-0.14.0
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CLOUD_CLIENT=OFF -DBUILD_EXAMPLES=OFF ..
sudo make install

#move back to plugin directory
cd ../../../../../

# build plugins and install to mujoco python installation directory
PYTHON_INSTALL_LOCATION=$(uv pip show mujoco | grep Location: | cut -d ' ' -f 2)
MUJOCO_ROOT_DIR="${PYTHON_INSTALL_LOCATION}/mujoco"

message="\n\n\nBuilding and installing plugins to mujoco python installation directory: ${MUJOCO_ROOT_DIR}\n\n\n"
echo -e $message

# optionally download c++ version of mujoco 
MUJOCO_VERSION=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('mujoco'))")


# Download the file
rm -rf precompiled
mkdir precompiled
cd precompiled
echo "Downloading precompiled mujoco version ${MUJOCO_VERSION}"
curl -L -o "mujoco-${MUJOCO_VERSION}.tar.gz" https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz

# Extract the tar.gz file
tar -xzf "mujoco-${MUJOCO_VERSION}.tar.gz"

# clean up
rm "mujoco-${MUJOCO_VERSION}.tar.gz"


PRECOMPILED_MUJOCO_DIR="${PWD}/mujoco-${MUJOCO_VERSION}"
echo "Precompiled_MUJOCO_DIR: ${PRECOMPILED_MUJOCO_DIR}"

cd ..
rm -rf ./plugin/build
mkdir -p ./plugin/build
cd ./plugin/build
cmake .. -DMUJOCO_ROOT_DIR=${MUJOCO_ROOT_DIR} -DCMAKE_BUILD_TYPE=Release -DPRECOMPILED_MUJOCO_DIR=${PRECOMPILED_MUJOCO_DIR}
make install

# force updating of system libraries  to see our manually installed stuff
sudo ldconfig

# force regeneration of xml model on install
cd ../../
echo "Regenerating xml model"
rm -f "./src/baloo_mujoco_sim/assets/"*.xml

#generate xml for use now?
python -c "import baloo_mujoco_sim"