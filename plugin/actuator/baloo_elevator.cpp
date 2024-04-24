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

#include "quintic.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::plugin::actuator
{
  namespace
  {

    constexpr char kAttrPGain[] = "kp";
    // constexpr char kAttrIGain[] = "ki";
    // constexpr char kAttrDGain[] = "kd";
    // constexpr char kAttrIMax[] = "imax";
    // constexpr char kAttrSlewMax[] = "slewmax";

    constexpr char kAttrAvgVel[] = "avgvel";

    std::optional<mjtNum> ReadOptionalDoubleAttr(const mjModel *m, int instance,
                                                 const char *attr)
    {
      const char *value = mj_getPluginConfig(m, instance, attr);
      if (value == nullptr || value[0] == '\0')
      {
        return std::nullopt;
      }
      return std::strtod(value, nullptr);
    }

    // returns the next act given the current act_dot, after clamping, for native
    // mujoco dyntypes.
    // copied from engine_forward.
    mjtNum NextActivation(const mjModel *m, const mjData *d, int actuator_id,
                          int act_adr, mjtNum act_dot)
    {
      mjtNum act = d->act[act_adr];

      if (m->actuator_dyntype[actuator_id] == mjDYN_FILTEREXACT)
      {
        // exact filter integration
        // act_dot(0) = (ctrl-act(0)) / tau
        // act(h) = act(0) + (ctrl-act(0)) (1 - exp(-h / tau))
        //        = act(0) + act_dot(0) * tau * (1 - exp(-h / tau))
        mjtNum tau = mju_max(mjMINVAL, m->actuator_dynprm[actuator_id * mjNDYN]);
        act = act + act_dot * tau * (1 - mju_exp(-m->opt.timestep / tau));
      }
      else
      {
        // Euler integration
        act = act + act_dot * m->opt.timestep;
      }

      // clamp to actrange
      if (m->actuator_actlimited[actuator_id])
      {
        mjtNum *actrange = m->actuator_actrange + 2 * actuator_id;
        act = mju_clip(act, actrange[0], actrange[1]);
      }

      return act;
    }

    // bool HasSlew(const mjModel* m, int instance) {
    //   return ReadOptionalDoubleAttr(m, instance, kAttrSlewMax).has_value();
    // }

  } // namespace

  QuinticConfig QuinticConfig::FromModel(const mjModel *m, int instance)
  {
    QuinticConfig config;
    config.p_gain = ReadOptionalDoubleAttr(m, instance, kAttrPGain).value_or(0);
    // config.i_gain = ReadOptionalDoubleAttr(m, instance, kAttrIGain).value_or(0);
    // config.d_gain = ReadOptionalDoubleAttr(m, instance, kAttrDGain).value_or(0);

    // ! user needs to define avgvel in m/s in xml file
    config.avgvel = ReadOptionalDoubleAttr(m, instance, kAttrAvgVel).value_or(0);

    // // Clamps in the XML are specified in terms of maximum forces. Scale by i_gain
    // // to get the limits on the value of the error integral.
    // std::optional<double> i_clamp_max_force =
    //     ReadOptionalDoubleAttr(m, instance, kAttrIMax);
    // if (i_clamp_max_force.has_value() && config.i_gain) {
    //   config.i_max = *i_clamp_max_force / config.i_gain;
    // }

    // config.slew_max = ReadOptionalDoubleAttr(m, instance, kAttrSlewMax);

    return config;
  }

  std::unique_ptr<Quintic> Quintic::Create(const mjModel *m, int instance)
  {
    QuinticConfig config = QuinticConfig::FromModel(m, instance);

    // if (config.i_max.has_value() && *config.i_max < 0) {
    //   mju_warning("negative imax");
    //   return nullptr;
    // }

    // if (config.slew_max.value_or(0.0) < 0) {
    //   mju_warning("maxslew must be non-negative");
    //   return nullptr;
    // }

    int actuator_idx = -1;
    for (int i = 0; i < m->nu; i++)
    {
      if (m->actuator_plugin[i] == instance)
      {
        if (actuator_idx != -1)
        {
          mju_warning("multiple actuators found for plugin instance %d",
                      instance);
          return nullptr;
        }
        actuator_idx = i;
      }
    }
    if (actuator_idx == -1)
    {
      mju_warning("actuator not found for plugin instance %d", instance);
      return nullptr;
    }
    return std::unique_ptr<Quintic>(new Quintic(config, actuator_idx));
  }

  void Quintic::Reset(mjtNum *plugin_state)
  {
    integral_ = 0.0;
    previous_ctrl_ = 0.0;
  }

  // mjtNum Quintic::GetCtrl(const mjModel* m, const mjData* d, const State& state,
  //                     bool actearly) const {
  //   mjtNum ctrl = 0;
  //   if (m->actuator_dyntype[actuator_idx_] == mjDYN_NONE) {
  //     ctrl = d->ctrl[actuator_idx_];
  //     // clamp ctrl
  //     if (m->actuator_ctrllimited[actuator_idx_]) {
  //       ctrl = mju_clip(ctrl, m->actuator_ctrlrange[2 * actuator_idx_],
  //                       m->actuator_ctrlrange[2 * actuator_idx_ + 1]);
  //     }
  //   } else {
  //     // Use of act instead of ctrl, to create integrated-velocity controllers or
  //     // to filter the controls.
  //     int actadr = m->actuator_actadr[actuator_idx_] +
  //                  m->actuator_actnum[actuator_idx_] - 1;
  //     if (actearly) {
  //       ctrl = NextActivation(m, d, actuator_idx_, actadr, d->act_dot[actadr]);
  //     } else {
  //       ctrl = d->act[actadr];
  //     }
  //   }
  //   if (config_.slew_max.has_value() && state.previous_ctrl_exists) {
  //     mjtNum ctrl_min = state.previous_ctrl - *config_.slew_max * m->opt.timestep;
  //     mjtNum ctrl_max = state.previous_ctrl + *config_.slew_max * m->opt.timestep;
  //     ctrl = mju_clip(ctrl, ctrl_min, ctrl_max);
  //   }
  //   return ctrl;
  // }

  // void Quintic::ActDot(const mjModel* m, mjData* d, int instance) const {
  //   State state = GetState(m, d, instance);
  //   mjtNum ctrl = GetCtrl(m, d, state, /*actearly=*/false);
  //   mjtNum error = ctrl - d->actuator_length[actuator_idx_];

  //   int state_idx = m->actuator_actadr[actuator_idx_];
  //   if (config_.i_gain) {
  //     mjtNum integral = state.integral + error * m->opt.timestep;
  //     if (config_.i_max.has_value()) {
  //       integral = mju_clip(integral, -*config_.i_max, *config_.i_max);
  //     }
  //     d->act_dot[state_idx] = (integral - d->act[state_idx]) / m->opt.timestep;
  //     ++state_idx;
  //   }
  //   if (config_.slew_max.has_value()) {
  //     d->act_dot[state_idx] = (ctrl - d->act[state_idx]) / m->opt.timestep;
  //     ++state_idx;
  //   }
  // }

  void Quintic::Compute(const mjModel *m, mjData *d, int instance)
  {

    mjtNum t = d->time;
    // initial conditions
    // need to get current velocity, pos, and accel of the joint where the actuator is attached
    // todo: I don't think this indexing with IDs is correct
    mjtNum v0 = d->qvel[m->actuator_trnid[actuator_idx_]]; //? should this be actuator velocity or joint velocity? not sure. They seem to use actuator velocity and actuator position
    mjtNum q0 = d->qpos[m->actuator_trnid[actuator_idx_]];
    mjtNum a0 = d->qacc[m->actuator_trnid[actuator_idx_]];

    // final conditions: stopped at the desired position
    mjtNum af = 0;
    mjtNum vf = 0;
    mjtNum qf = d->ctrl[actuator_idx_];

    // if control is different than previous control, replan trajectory
    if (qf != previous_ctrl_)
    {
      mjtNum t0 = d->time;
      tf_ = mju_abs(qf - q0) / config_.avgvel + t0;

      // form T matrix
      Eigen::Matrix<mjtNum, 6, 6> T;
      T << 1, t0, t0 * t0, t0 * t0 * t0, t0 * t0 * t0 * t0, t0 * t0 * t0 * t0 * t0,
          0, 1, 2 * t0, 3 * t0 * t0, 4 * t0 * t0 * t0, 5 * t0 * t0 * t0 * t0,
          0, 0, 2, 6 * t0, 12 * t0 * t0, 20 * t0 * t0 * t0,
          1, tf_, tf_ * tf_, tf_ * tf_ * tf_, tf_ * tf_ * tf_ * tf_, tf_ * tf_ * tf_ * tf_ * tf_,
          0, 1, 2 * tf_, 3 * tf_ * tf_, 4 * tf_ * tf_ * tf_, 5 * tf_ * tf_ * tf_ * tf_,
          0, 0, 2, 6 * tf_, 12 * tf_ * tf_, 20 * tf_ * tf_ * tf_;

      // form q vector
      Eigen::Vector<mjtNum, 6> q;
      q << q0, v0, a0, qf, vf, af;

      // solve for coeff vector
      Eigen::Vector c = T.fullPivLu().solve(q);
      std::copy(c.data(), c.data() + c.size(), coeffs_);

      // update delayed vars
      previous_ctrl_ = qf;
    }

    // compute quintic velocity trajectory
    mjtNum qdes = coeffs_[0] + coeffs_[1] * t + coeffs_[2] * t * t + coeffs_[3] * t * t * t + coeffs_[4] * t * t * t * t + coeffs_[5] * t * t * t * t * t;
    mjtNum vdes = coeffs_[1] + 2 * coeffs_[2] * t + 3 * coeffs_[3] * t * t + 4 * coeffs_[4] * t * t * t + 5 * coeffs_[5] * t * t * t * t;
    mjtNum ades = 2 * coeffs_[2] + 6 * coeffs_[3] * t + 12 * coeffs_[4] * t * t + 20 * coeffs_[5] * t * t * t;

    mjtNum velError = vdes - v0;
    // ! the issues is that I have actuator pos and vel, but not acceleration. And I need that for this trajectory planner....

    d->actuator_force[actuator_idx_] = config_.p_gain * velError

    // State state = GetState(m, d, instance);
    // mjtNum ctrl = GetCtrl(m, d, state, m->actuator_actearly[actuator_idx_]);

    // mjtNum error = ctrl - d->actuator_length[actuator_idx_];

    // mjtNum ctrl_dot = m->actuator_dyntype[actuator_idx_] == mjDYN_NONE
    //                       ? 0
    //                       : d->act_dot[m->actuator_actadr[actuator_idx_] +
    //                                    m->actuator_actnum[actuator_idx_] - 1];
    // mjtNum error_dot = ctrl_dot - d->actuator_velocity[actuator_idx_];

    // if (config_.i_gain) {
    //   integral_ = state.integral + error * m->opt.timestep;
    //   if (config_.i_max.has_value()) {
    //     integral_ = mju_clip(integral_, -*config_.i_max, *config_.i_max);
    //   }
    // }

    // d->actuator_force[actuator_idx_] = config_.p_gain * error +
    //                                    config_.d_gain * error_dot +
    //                                    config_.i_gain * integral_;
    // previous_ctrl_ = ctrl;
  }

  void Quintic::Advance(const mjModel *m, mjData *d, int instance) const
  {
    // act variables already updated by MuJoCo integrating act_dot
  }

  int Quintic::StateSize(const mjModel *m, int instance)
  {
    return 0;
  }

  int Quintic::ActDim(const mjModel *m, int instance, int actuator_id)
  {
    // double i_gain = ReadOptionalDoubleAttr(m, instance, kAttrIGain).value_or(0);
    // return (i_gain ? 1 : 0) + (HasSlew(m, instance) ? 1 : 0);

    // my
    return 0;
  }

  // Quintic::State Quintic::GetState(const mjModel* m, mjData* d, int instance) const {
  //   State state;
  //   int state_idx = m->actuator_actadr[instance];
  //   if (config_.i_gain) {
  //     state.integral = d->act[state_idx++];
  //   }
  //   if (config_.slew_max.has_value()) {
  //     state.previous_ctrl = d->act[state_idx++];
  //     state.previous_ctrl_exists = d->time > 0;
  //   }
  //   return state;
  // }

  void Quintic::RegisterPlugin()
  {
    mjpPlugin plugin;
    mjp_defaultPlugin(&plugin);
    plugin.name = "mujoco.quintic";
    plugin.capabilityflags |= mjPLUGIN_ACTUATOR;

    // std::vector<const char*> attributes = {kAttrPGain, kAttrIGain, kAttrDGain,
    //  kAttrIMax, kAttrSlewMax};
    std::vector<const char *> attributes = {kAttrPGain, kAttrAvgVel};

    plugin.nattribute = attributes.size();
    plugin.attributes = attributes.data();

    // plugin.actuator_actdim = Quintic::ActDim;
    plugin.nstate = Quintic::StateSize;

    plugin.init = +[](const mjModel *m, mjData *d, int instance)
    {
      std::unique_ptr<Quintic> pid = Quintic::Create(m, instance);
      if (pid == nullptr)
      {
        return -1;
      }
      d->plugin_data[instance] = reinterpret_cast<uintptr_t>(pid.release());
      return 0;
    };
    plugin.destroy = +[](mjData *d, int instance)
    {
      delete reinterpret_cast<Quintic *>(d->plugin_data[instance]);
      d->plugin_data[instance] = 0;
    };
    plugin.reset = +[](const mjModel *m, double *plugin_state, void *plugin_data,
                       int instance)
    {
      auto *pid = reinterpret_cast<Quintic *>(plugin_data);
      pid->Reset(plugin_state);
    };
    // plugin.actuator_act_dot = +[](const mjModel* m, mjData* d, int instance) {
    //   auto* pid = reinterpret_cast<Quintic*>(d->plugin_data[instance]);
    //   pid->ActDot(m, d, instance);
    // };
    plugin.compute =
        +[](const mjModel *m, mjData *d, int instance, int capability_bit)
    {
      auto *pid = reinterpret_cast<Quintic *>(d->plugin_data[instance]);
      pid->Compute(m, d, instance);
    };
    plugin.advance = +[](const mjModel *m, mjData *d, int instance)
    {
      auto *pid = reinterpret_cast<Quintic *>(d->plugin_data[instance]);
      pid->Advance(m, d, instance);
    };
    // TODO: b/303823996 - allow actuator plugins to compute their derivatives wrt
    // qvel, for implicit integration
    mjp_registerPlugin(&plugin);
  }

  Quintic::Quintic(QuinticConfig config, int actuator_idx)
      : config_(std::move(config)), actuator_idx_(actuator_idx) {}

} // namespace mujoco::plugin::actuator
