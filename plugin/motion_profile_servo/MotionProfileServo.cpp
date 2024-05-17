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

#include "MotionProfileServo.hpp"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::plugin::actuator {
  namespace {

    constexpr char kAttrPGain[] = "kp";
    constexpr char kAttrVGain[] = "kv";
    constexpr char kAttrZeta[] = "zeta";
    constexpr char kAttrWn[] = "omega_n";

    std::optional<mjtNum> ReadOptionalDoubleAttr(const mjModel* m, int instance,
      const char* attr) {
      const char* value = mj_getPluginConfig(m, instance, attr);
      if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
      }
      return std::strtod(value, nullptr);
    }

    // returns the next act given the current act_dot, after clamping, for native
    // mujoco dyntypes.
    // copied from engine_forward.
    mjtNum NextActivation(const mjModel* m, const mjData* d, int actuator_id,
      int act_adr, mjtNum act_dot) {
      mjtNum act = d->act[act_adr];

      if (m->actuator_dyntype[actuator_id] == mjDYN_FILTEREXACT) {
        // exact filter integration
        // act_dot(0) = (ctrl-act(0)) / tau
        // act(h) = act(0) + (ctrl-act(0)) (1 - exp(-h / tau))
        //        = act(0) + act_dot(0) * tau * (1 - exp(-h / tau))
        mjtNum tau = mju_max(mjMINVAL, m->actuator_dynprm[actuator_id * mjNDYN]);
        act = act + act_dot * tau * (1 - mju_exp(-m->opt.timestep / tau));
      }
      else {
        // Euler integration
        act = act + act_dot * m->opt.timestep;
      }

      // clamp to actrange
      if (m->actuator_actlimited[actuator_id]) {
        mjtNum* actrange = m->actuator_actrange + 2 * actuator_id;
        act = mju_clip(act, actrange[0], actrange[1]);
      }

      return act;
    }

  }  // namespace

  ServoConfig ServoConfig::FromModel(const mjModel* m, int instance) {
    ServoConfig config;
    config.p_gain = ReadOptionalDoubleAttr(m, instance, kAttrPGain).value_or(0);
    config.v_gain = ReadOptionalDoubleAttr(m, instance, kAttrVGain).value_or(0);
    config.zeta = ReadOptionalDoubleAttr(m, instance, kAttrZeta).value_or(0);
    config.wn = ReadOptionalDoubleAttr(m, instance, kAttrWn).value_or(0);

    return config;
  }

  std::unique_ptr<MotionProfileServo> MotionProfileServo::Create(const mjModel* m, int instance) {
    ServoConfig config = ServoConfig::FromModel(m, instance);

    //check config for invalid values
    if (config.p_gain < 0) {
      mju_warning("MotionProfileServo plugin: negative pgain");
      return nullptr;
    }

    if (config.v_gain < 0) {
      mju_warning("MotionProfileServo plugin: negative vgain");
      return nullptr;
    }

    if (config.zeta <= 0) {
      mju_warning("MotionProfileServo plugin: non-positive zeta");
      return nullptr;
    }

    if (config.wn <= 0) {
      mju_warning("MotionProfileServo plugin: non-positive wn");
      return nullptr;
    }

    // loop through number of inputs and find actuators that are controlled by this
    std::vector<int> actuators;
    for (int i = 0; i < m->nu; i++) {
      if (m->actuator_plugin[i] == instance) {
        actuators.push_back(i);
      }
    }

    if (actuators.empty()) {
      mju_warning("actuator not found for plugin instance %d", instance);
      return nullptr;
    }
    return std::unique_ptr<MotionProfileServo>(new MotionProfileServo(config, std::move(actuators)));
  }

  void MotionProfileServo::Reset(mjtNum* plugin_state) {}

  mjtNum MotionProfileServo::GetCtrl(
    const mjModel* m,
    const mjData* d,
    int actuator_idx,
    const State& state,
    bool actearly
  ) const {

    mjtNum ctrl = 0;

    //?should it even be an option to NOT have a ctrl limit? (the ctrl signal is position command)
    if (m->actuator_dyntype[actuator_idx] == mjDYN_NONE) {
      ctrl = d->ctrl[actuator_idx];
      // clamp ctrl
      if (m->actuator_ctrllimited[actuator_idx]) {
        ctrl = mju_clip(ctrl, m->actuator_ctrlrange[2 * actuator_idx],
          m->actuator_ctrlrange[2 * actuator_idx + 1]);
      }
    }

    return ctrl / 1000; // since I want ctrl signal to be in mm to match hardware
  }

  MotionProfileServo::StateDot MotionProfileServo::GetPrefilterStateDot(
    const mjModel* m,
    const mjData* d,
    int actuator_idx,
    mjtNum ctrl, // should come in in meters
    const State& state
  ) const {
    StateDot stateDot;


    // second order system parameterized by zeta and wn.
    double zeta = config_.zeta;
    double wn = config_.wn;
    stateDot.prefiltered_position_command_ddot =
      -2 * zeta * wn * state.prefiltered_position_command_dot
      - wn * wn * state.prefiltered_position_command
      + wn * wn * ctrl;

    stateDot.prefiltered_position_command_dot = state.prefiltered_position_command_dot;

    return stateDot;
  }

  void MotionProfileServo::ActDot(const mjModel* m, mjData* d, int instance) const {
    for (int actuator_idx : actuators_) {
      State prefilterState = GetPrefilterState(m, d, actuator_idx);
      mjtNum ctrl = GetCtrl(m, d, actuator_idx, prefilterState, /*actearly=*/false);

      //prefilter to calculate the smoothed position command I want to follow.
      StateDot prefilterStateDot = GetPrefilterStateDot(m, d, actuator_idx, ctrl, prefilterState);

      int state_idx = m->actuator_actadr[actuator_idx];

      //assumes order in act_dot array is [xdot, x] for activation
      d->act_dot[state_idx++] = prefilterStateDot.prefiltered_position_command_ddot;
      d->act_dot[state_idx] = prefilterStateDot.prefiltered_position_command_dot;
    }
  }

  void MotionProfileServo::Compute(const mjModel* m, mjData* d, int instance) {
    for (int i = 0; i < actuators_.size(); i++) {
      int actuator_idx = actuators_[i];
      State prefilterState = GetPrefilterState(m, d, actuator_idx);
      mjtNum ctrl = GetCtrl(m, d, actuator_idx, prefilterState, m->actuator_actearly[actuator_idx]);

      mjtNum position_error = prefilterState.prefiltered_position_command - d->actuator_length[actuator_idx];

      mjtNum velocity_cmd = config_.v_gain * (position_error);

      mjtNum velocity_error = velocity_cmd - d->actuator_velocity[actuator_idx];

      mjtNum force = config_.v_gain * velocity_error;

      d->actuator_force[actuator_idx] = force;
    }
  }

  void MotionProfileServo::Advance(const mjModel* m, mjData* d, int instance) const {
    // act variables already updated by MuJoCo integrating act_dot
  }

  int MotionProfileServo::StateSize(const mjModel* m, int instance) {
    return 0;
  }

  int MotionProfileServo::ActDim(const mjModel* m, int instance, int actuator_id) {
    return 2; // filtered command and filtered command derivative since its a second order prefilter.
  }

  MotionProfileServo::State MotionProfileServo::GetPrefilterState(const mjModel* m, mjData* d, int actuator_idx) const {
    State state;
    //get address in activation array
    int state_idx = m->actuator_actadr[actuator_idx];

    // activation assumed to be in [xdot, x] order
    state.prefiltered_position_command_dot = d->act[state_idx++];
    state.prefiltered_position_command = d->act[state_idx];

    return state;
  }

  void MotionProfileServo::RegisterPlugin() {
    mjpPlugin plugin;
    mjp_defaultPlugin(&plugin);
    plugin.name = "mujoco.actuator.motion_profile_servo";
    plugin.capabilityflags |= mjPLUGIN_ACTUATOR;

    std::vector<const char*> attributes = { kAttrPGain, kAttrVGain, kAttrZeta, kAttrWn };
    plugin.nattribute = attributes.size();
    plugin.attributes = attributes.data();

    plugin.actuator_actdim = MotionProfileServo::ActDim;
    plugin.nstate = MotionProfileServo::StateSize;

    plugin.init = +[](const mjModel* m, mjData* d, int instance) {
      std::unique_ptr<MotionProfileServo> motion_profile_servo = MotionProfileServo::Create(m, instance);
      if (motion_profile_servo == nullptr) {
        return -1;
      }
      d->plugin_data[instance] = reinterpret_cast<uintptr_t>(motion_profile_servo.release());
      return 0;
      };

    plugin.destroy = +[](mjData* d, int instance) {
      delete reinterpret_cast<MotionProfileServo*>(d->plugin_data[instance]);
      d->plugin_data[instance] = 0;
      };

    plugin.reset = +[](const mjModel* m, double* plugin_state, void* plugin_data,
      int instance) {
        auto* motion_profile_servo = reinterpret_cast<MotionProfileServo*>(plugin_data);
        motion_profile_servo->Reset(plugin_state);
      };

    plugin.actuator_act_dot = +[](const mjModel* m, mjData* d, int instance) {
      auto* motion_profile_servo = reinterpret_cast<MotionProfileServo*>(d->plugin_data[instance]);
      motion_profile_servo->ActDot(m, d, instance);
      };

    plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
      auto* motion_profile_servo = reinterpret_cast<MotionProfileServo*>(d->plugin_data[instance]);
      motion_profile_servo->Compute(m, d, instance);
      };

    plugin.advance = +[](const mjModel* m, mjData* d, int instance) {
      auto* motion_profile_servo = reinterpret_cast<MotionProfileServo*>(d->plugin_data[instance]);
      motion_profile_servo->Advance(m, d, instance);
      };

    // TODO: b/303823996 - allow actuator plugins to compute their derivatives wrt
    // qvel, for implicit integration
    mjp_registerPlugin(&plugin);
  }

  MotionProfileServo::MotionProfileServo(ServoConfig config, std::vector<int> actuators)
    : config_(std::move(config)), actuators_(std::move(actuators)) {}

}  // namespace mujoco::plugin::actuator