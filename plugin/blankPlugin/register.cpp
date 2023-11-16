#include <mujoco/mjplugin.h>
#include "blank_plugin.hpp"

namespace mujoco::plugin::sensor {

mjPLUGIN_LIB_INIT { Blank::RegisterPlugin(); }

}  // namespace mujoco::plugin::sensor