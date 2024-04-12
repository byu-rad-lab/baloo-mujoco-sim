// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_PLUGIN_ACTUATOR_QUINTIC_H_
#define MUJOCO_PLUGIN_ACTUATOR_QUINTIC_H_

#include <memory>
#include <optional>
#include <Eigen/Dense>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>

namespace mujoco::plugin::actuator {

struct QuinticConfig {
  // double p_gain = 0.0;
  // double i_gain = 0.0;
  // double d_gain = 0.0;

  double p_gain = 0.0;

  // average velocity of actuator. Heuristic to limit velocities and accelerations to plausible levels.
  double avgvel = 0.0;

  // // Maximum value of the error integral.
  // // NOTE: In the XML definition, the clamp values are specified as limits on
  // // the *force* value, so are scaled by i_gain.
  // std::optional<double> i_max;

  // // Maximum speed at which the setpoint can change.
  // std::optional<double> slew_max;

  // Reads plugin attributes to construct Qiuntic configuration.
  static QuinticConfig FromModel(const mjModel* m, int instance);
};

// An actuator plugin which implements configurable Qiuntic control.
class Quintic {
 public:
  // Returns an instance of Qiuntic. The result can be null in case of
  // misconfiguration.
  static std::unique_ptr<Quintic> Create(const mjModel* m, int instance);

  // Returns the number of state variables for the plugin instance
  static int StateSize(const mjModel* m, int instance);

  // Returns the number of activation variables for the plugin instance
  static int ActDim(const mjModel* m, int instance, int actuator_id);

  // Resets the C++ Qiuntic instance's state.
  // plugin_state is a C array pointer into mjData->plugin_state, with a size
  // equal to the value returned from StateSize.
  void Reset(mjtNum* plugin_state);

  // // Computes the rate of change for activation variables
  // void ActDot(const mjModel* m, mjData* d, int instance) const;

  // Idempotent computation which updates d->actuator_force and the internal
  // state of the class. Called after ActDot.
  void Compute(const mjModel* m, mjData* d, int instance);

  // Updates plugin state.
  void Advance(const mjModel* m, mjData* d, int instance) const;

  // Adds the Qiuntic plugin to the global registry of MuJoCo plugins.
  static void RegisterPlugin();

 private:
  Quintic(QuinticConfig config, int actuator_idx);

  // struct State {
  //   mjtNum previous_ctrl = 0;
  //   // if using slew rate limits, mjData.act will contain an activation variable
  //   // with the last ctrl value. If `false`, that value should be ignored,
  //   // because it hasn't been set yet.
  //   bool previous_ctrl_exists = false;
  //   mjtNum integral = 0;
  // };
  // // Reads data from d->act and returns it as a State struct.
  // State GetState(const mjModel* m, mjData* d, int instance) const;

  // // Returns the Qiuntic setpoint, which is normally d->ctrl, but can be d->act for
  // // actuators with dyntype != none.
  // mjtNum GetCtrl(const mjModel* m, const mjData* d, const State& state,
  //                bool actearly) const;

  QuinticConfig config_;
  int actuator_idx_ = 0;

  mjtNum integral_ = 0.0;
  mjtNum previous_ctrl_ = 0.0;
  mjtNum tf_ = 0.0;
  mjtNum coeffs_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

}  // namespace mujoco::plugin::actuator

#endif  // MUJOCO_PLUGIN_ACTUATOR_QUINTIC_H_
