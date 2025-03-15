/*
Curtis Johnson 2025
This plugin implements a Ruckig-based actuator controller for MuJoCo.

See https://github.com/pantor/ruckig for more information about Ruckig.

The user can specify the following attributes in the plugin element:
- max_velocity: the maximum velocity of the actuator.
- max_acceleration: the maximum acceleration of the actuator.
- max_jerk: the maximum jerk of the actuator.

The 'activations' of this plugin are defined as:

mjData.act = [current_velocity, current_position]
mjData.act_dot = [current_accleration, current_velocity]

Note that the actual locations of these values in mjData.act and mjData.act_dot
are tracked by Mujoco.
*/

#include "RuckigActuator.hpp"

#include <mujoco/mujoco.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mujoco::plugin::actuator
{
    namespace
    {

        constexpr char kMaxVel[] = "max_velocity";
        constexpr char kMaxAccel[] = "max_acceleration";
        constexpr char kMaxJerk[] = "max_jerk";
        constexpr char kKp[] = "kp";
        constexpr char kKv[] = "kv";
        constexpr char kKa[] = "ka";

        std::optional<mjtNum> ReadOptionalDoubleAttr(const mjModel* m, int instance, const char* attr)
        {
            const char* value = mj_getPluginConfig(m, instance, attr);
            if (value == nullptr || value[0] == '\0')
            {
                return std::nullopt;
            }
            return std::strtod(value, nullptr);
        }

        // returns the next act given the current act_dot, after clamping, for native
        // mujoco dyntypes.
        // copied from engine_forward.
        mjtNum NextActivation(const mjModel* m, const mjData* d, int actuator_id,
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
                mjtNum* actrange = m->actuator_actrange + 2 * actuator_id;
                act = mju_clip(act, actrange[0], actrange[1]);
            }

            return act;
        }

    } // namespace

    RuckigConfig RuckigConfig::FromModel(const mjModel* m, int instance)
    {
        RuckigConfig config;
        config.dt = m->opt.timestep;
        config.max_velocity = ReadOptionalDoubleAttr(m, instance, kMaxVel).value_or(0);
        config.max_acceleration = ReadOptionalDoubleAttr(m, instance, kMaxAccel).value_or(0);
        config.max_jerk = ReadOptionalDoubleAttr(m, instance, kMaxJerk).value_or(0);

        config.kp = ReadOptionalDoubleAttr(m, instance, kKp).value_or(0);
        config.kv = ReadOptionalDoubleAttr(m, instance, kKv).value_or(0);
        config.ka = ReadOptionalDoubleAttr(m, instance, kKa).value_or(0);

        return config;
    }

    std::unique_ptr<RuckigActuator> RuckigActuator::Create(const mjModel* m, int instance)
    {
        RuckigConfig config = RuckigConfig::FromModel(m, instance);

        // check config for invalid values
        if (config.max_velocity <= 0)
        {
            mju_warning("RuckigActuator plugin: non-positive max_velocity");
            return nullptr;
        }

        if (config.max_acceleration <= 0)
        {
            mju_warning("RuckigActuator plugin: non-positive max_acceleration");
            return nullptr;
        }

        if (config.max_jerk <= 0)
        {
            mju_warning("RuckigActuator plugin: non-positive max_jerk");
            return nullptr;
        }

        if (config.dt <= 0)
        {
            mju_warning("RuckigActuator plugin: non-positive timestep");
            return nullptr;
        }

        if (config.kp < 0)
        {
            mju_warning("RuckigActuator plugin: negative kp");
            return nullptr;
        }

        if (config.kv < 0)
        {
            mju_warning("RuckigActuator plugin: negative kv");
            return nullptr;
        }

        if (config.ka < 0)
        {
            mju_warning("RuckigActuator plugin: negative ka");
            return nullptr;
        }

        // loop through number of inputs and find actuators that are controlled by this
        std::vector<int> actuators;
        for (int i = 0; i < m->nu; i++)
        {
            if (m->actuator_plugin[i] == instance)
            {
                actuators.push_back(i);
            }
        }

        if (actuators.empty())
        {
            mju_warning("actuator not found for plugin instance %d", instance);
            return nullptr;
        }
        // Validate actnum values for all actuators:
        for (int actuator_id : actuators)
        {
            int actnum = m->actuator_actnum[actuator_id];
            int expected_actnum = RuckigActuator::ActDim(m, instance, actuator_id);
            int dyntype = m->actuator_dyntype[actuator_id];
            if (dyntype == mjDYN_FILTER || dyntype == mjDYN_FILTEREXACT || dyntype == mjDYN_INTEGRATOR)
            {
                expected_actnum++;
            }
            if (actnum != expected_actnum)
            {
                mju_warning("actuator %d has actdim %d, expected %d. Add "
                    "actdim=\"%d\" to the "
                    "actuator plugin element.",
                    actuator_id, actnum, expected_actnum, expected_actnum);
                return nullptr;
            }
        }

        return std::unique_ptr<RuckigActuator>(
            new RuckigActuator(config, std::move(actuators)));
    }

    void RuckigActuator::Reset(mjtNum* plugin_state) {

        for (int actuator_idx : actuators_) {
            ConfigureRuckig();
        }

    }

    mjtNum RuckigActuator::GetCtrl(
        const mjModel* m,
        const mjData* d,
        int actuator_idx,
        bool actearly) const
    {
        mjtNum ctrl = 0;

        //?should it even be an option to NOT have a ctrl limit? 
        // (the ctrl signal is position command)
        if (m->actuator_dyntype[actuator_idx] == mjDYN_NONE)
        {
            ctrl = d->ctrl[actuator_idx];
            // clamp ctrl
            if (m->actuator_ctrllimited[actuator_idx])
            {
                ctrl = mju_clip(ctrl, m->actuator_ctrlrange[2 * actuator_idx],
                    m->actuator_ctrlrange[2 * actuator_idx + 1]);
            }
        }

        return ctrl / 1000; // gui ctrl is in mm, convert to meters. Consider changing this.
    }

    void RuckigActuator::ActDot(const mjModel* m, mjData* d, int instance)
    {
        for (int actuator_idx : actuators_)
        {
            mjtNum position_cmd = GetCtrl(m, d, actuator_idx, /*actearly=*/false);
            input_param_.target_position = { position_cmd };

            // prefilter to calculate the smoothed position command I want to follow.
            trajectory_planner_.update(input_param_, output_param_);

            int state_idx = m->actuator_actadr[actuator_idx];

            //fill xdot = [a, v] in act_dot
            d->act_dot[state_idx++] = output_param_.new_acceleration[0];
            d->act_dot[state_idx] = output_param_.new_velocity[0];

            // DONT CALL pass_to_input here. This is just for mujoco to finite difference the act_dot.
            // see https://mujoco.readthedocs.io/en/3.2.3/programming/extension.html#actuator-activations
            // mjpPlugin.advance will be called after this is integrated which will override the act_dot integration.
        }
    }

    void RuckigActuator::Compute(const mjModel* m, mjData* d, int instance)
    {

        for (int actuator_idx : actuators_)
        {
            //get the current position, velocity, and acceleration from the ruckig object
            mjtNum desired_position = output_param_.new_position[0];
            mjtNum desired_velocity = output_param_.new_velocity[0];
            mjtNum desired_acceleration = output_param_.new_acceleration[0];

            //convert to forces using PD + FF control
            mjtNum position_error = desired_position - d->actuator_length[actuator_idx];
            mjtNum velocity_error = desired_velocity - d->actuator_velocity[actuator_idx];

            // cancel out as many dynamic terms as possible to simulate stepper holding torque.
            // Note that this does NOT include mouse perturbations, which are only applied through xfrc_applied.
            // so this plugin will not cancel those out. 
            // Future work can map mouse perturbations to generalized forces to cancel here. Should just need to get correct jacobians.
            // see https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework

            // get the id of the joint's dof
            int elevator_joint_id = mj_name2id(m, mjOBJ_JOINT, "linear_actuator");
            int dof_id = m->jnt_dofadr[elevator_joint_id];
            mjtNum qfrc_bias = d->qfrc_bias[dof_id];
            mjtNum qfrc_smooth = d->qfrc_smooth[dof_id];
            mjtNum qfrc_passive = d->qfrc_passive[dof_id];
            mjtNum qfrc_inverse = d->qfrc_inverse[dof_id];

            mjtNum qfrc_total = qfrc_bias + qfrc_smooth + qfrc_passive + qfrc_inverse;


            mjtNum force = qfrc_total +
                config_.kp * position_error +
                config_.kv * velocity_error +
                config_.ka * desired_acceleration;

            d->actuator_force[actuator_idx] = force;
        }
    }

    void RuckigActuator::Advance(const mjModel* m, mjData* d, int instance)
    {
        for (int actuator_idx : actuators_)
        {
            output_param_.pass_to_input(input_param_);
        }
    }

    int RuckigActuator::StateSize(const mjModel* m, int instance)
    {
        // see
        // https://mujoco.readthedocs.io/en/stable/programming/extension.html#actuator-states
        // I use actdim instead of state.
        return 0;
    }

    int RuckigActuator::ActDim(const mjModel* m, int instance, int actuator_id)
    {
        return 2; // [vel, pos]
    }


    void RuckigActuator::RegisterPlugin()
    {
        mjpPlugin plugin;
        mjp_defaultPlugin(&plugin);
        plugin.name = "mujoco.actuator.ruckig_actuator";
        plugin.capabilityflags |= mjPLUGIN_ACTUATOR;

        std::vector<const char*> attributes = { kMaxVel, kMaxAccel, kMaxJerk, kKp, kKv, kKa };
        plugin.nattribute = attributes.size();
        plugin.attributes = attributes.data();
        plugin.nstate = RuckigActuator::StateSize;

        plugin.init = +[](const mjModel* m, mjData* d, int instance)
            {
                std::unique_ptr<RuckigActuator> ruckig_actuator = RuckigActuator::Create(m, instance);
                if (ruckig_actuator == nullptr)
                {
                    return -1;
                }
                d->plugin_data[instance] = reinterpret_cast<uintptr_t>(ruckig_actuator.release());
                return 0;
            };

        plugin.destroy = +[](mjData* d, int instance)
            {
                delete reinterpret_cast<RuckigActuator*>(d->plugin_data[instance]);
                d->plugin_data[instance] = 0;
            };

        plugin.reset = +[](const mjModel* m, double* plugin_state,
            void* plugin_data, int instance)
            {
                auto* ruckig_actuator = reinterpret_cast<RuckigActuator*>(plugin_data);
                ruckig_actuator->Reset(plugin_state);
            };

        plugin.actuator_act_dot = +[](const mjModel* m, mjData* d, int instance)
            {
                auto* ruckig_actuator = reinterpret_cast<RuckigActuator*>(d->plugin_data[instance]);
                ruckig_actuator->ActDot(m, d, instance);
            };

        plugin.compute = +[](const mjModel* m, mjData* d, int instance, int capability_bit)
            {
                auto* ruckig_actuator = reinterpret_cast<RuckigActuator*>(d->plugin_data[instance]);
                ruckig_actuator->Compute(m, d, instance);
            };

        plugin.advance = +[](const mjModel* m, mjData* d, int instance)
            {
                auto* ruckig_actuator = reinterpret_cast<RuckigActuator*>(d->plugin_data[instance]);
                ruckig_actuator->Advance(m, d, instance);
            };

        // TODO: b/303823996 - allow actuator plugins to compute their derivatives
        // wrt qvel, for implicit integration
        mjp_registerPlugin(&plugin);
    }

    RuckigActuator::RuckigActuator(
        RuckigConfig config,
        std::vector<int> actuators) : config_(std::move(config)), actuators_(std::move(actuators)), trajectory_planner_(config_.dt)
    {
        ConfigureRuckig();
    }

    void RuckigActuator::ConfigureRuckig()
    {
        output_param_.new_position[0] = 0;
        output_param_.new_velocity[0] = 0;
        output_param_.new_acceleration[0] = 0;
        output_param_.new_jerk[0] = 0;
        output_param_.time = 0;
        //copy reset output to the input
        output_param_.pass_to_input(input_param_);

        input_param_.max_velocity = { config_.max_velocity };
        input_param_.max_acceleration = { config_.max_acceleration };
        input_param_.max_jerk = { config_.max_jerk };
        input_param_.target_position = { 0.0 };
        input_param_.target_velocity = { 0.0 };
        input_param_.target_acceleration = { 0.0 };
    }

} // namespace mujoco::plugin::actuator