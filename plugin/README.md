# Custom Plugins for Mujoco
This foldr contains a few custom [plugins](https://mujoco.readthedocs.io/en/latest/programming/extension.html) for Mujoco.

Plugins expand the functionality of mujoco by allowing users to write custom C code that can be called from the simulation. This is useful for implementing custom dynamics, sensors, or other features that are not available in the standard Mujoco library.

Users of ```baloo-mujoco-sim`` do not need to build these directly, since the will be built when running the included ```install.sh``` script in the root directory. 

For details on what each plugin does, see the README files in each of the subdirectories.