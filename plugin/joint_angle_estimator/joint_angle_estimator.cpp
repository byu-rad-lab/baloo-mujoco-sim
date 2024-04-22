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

#include "joint_angle_estimator.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include <iostream>

namespace mujoco::plugin::sensor {

    namespace {

        // Checks that a plugin config attribute exists.
        bool CheckAttr(const std::string& input) {
            char* end;
            std::string value = input;
            value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
            strtod(value.c_str(), &end);
            return end == value.data() + value.size();
        }


    }  // namespace

    // Creates a JointAngleEstimator instance if all config attributes are defined and
    // within their allowed bounds.
    JointAngleEstimator* JointAngleEstimator::Create(const mjModel* m, mjData* d, int instance) {
        if (CheckAttr(std::string(mj_getPluginConfig(m, instance, "nchannel")))) {
            // nchannel
            int nchannel = strtod(mj_getPluginConfig(m, instance, "nchannel"), nullptr);
            if (!nchannel) nchannel = 1;
            if (nchannel < 1 || nchannel > 6) {
                mju_error("nchannel must be between 1 and 6");
                return nullptr;
            }

            return new JointAngleEstimator(m, d, instance, nchannel);
        }
        else {
            mju_error("Invalid or missing parameters in blank sensor plugin");
            return nullptr;
        }
    }

    JointAngleEstimator::JointAngleEstimator(const mjModel* m, mjData* d, int instance, int nchannel) : nchannel_(nchannel)
    {
    }

    void JointAngleEstimator::Reset(const mjModel* m, int instance) {}

    void JointAngleEstimator::Compute(const mjModel* m, mjData* d, int instance) {
        mj_markStack(d);

        // Get sensor id to which this plugin is attached.
        int id;
        for (id = 0; id < m->nsensor; ++id) {
            if (m->sensor_type[id] == mjSENS_PLUGIN &&
                m->sensor_plugin[id] == instance) {
                break;
            }
        }

        // need to find a way to map instance or id to joint number
        const char* name = mj_id2name(m, mjOBJ_SENSOR, id);
        std::string sensorName(name);
        // mju_error("sensorName: %s", sensorName.c_str());

        // Clear sensordata
        mjtNum* mysensordata = d->sensordata + m->sensor_adr[id];
        mju_zero(mysensordata, m->sensor_dim[id]);

        //compute sensor data and fill appropriate sensordata array

        // FOR JOINT ANGLES
        //get orientation of first and last disk in global frame
        std::string numDisksName = "num_disks";
        int numDisksId = mj_name2id(m, mjOBJ_NUMERIC, numDisksName.c_str());
        if (numDisksId == -1)
        {
            mju_error("numDisksId is -1");
            return;
        }
        mjtNum* numDisks = m->numeric_data + numDisksId;

        // ?? how to deal with different jointnum (i.e. 1_B0 and 2_B0)?
        // f"{side}_{joint_num}_B{i}" is format of disk bodies generally
        std::string base_quat_name = sensorName + "_B0_framequat";
        int base_quat_id = mj_name2id(m, mjOBJ_SENSOR, base_quat_name.c_str());

        if (base_quat_id == -1)
        {
            mju_error("base_quat_id is -1");
            return;
        }

        //get sensor_adr out of model with sensor id
        int base_quat_adr = m->sensor_adr[base_quat_id];
        mjtNum* r_base = d->sensordata + base_quat_adr; //should be 1,0,0,0

        std::string tip_quat_name = sensorName + "_B" + std::to_string(int(*numDisks) - 1) + "_framequat";
        int tip_quat_id = mj_name2id(m, mjOBJ_SENSOR, tip_quat_name.c_str());
        if (tip_quat_id == -1)
        {
            mju_error("tip_quat_id is -1");
            return;
        }
        int tip_quat_adr = m->sensor_adr[tip_quat_id];
        mjtNum* r_tip = d->sensordata + tip_quat_adr;

        //get relative orientation between first and last disk
        mjtNum r_base_conj[4]{ 0 };
        mju_negQuat(r_base_conj, r_base);

        mjtNum r_base2Tip[4]{ 0 };
        mju_mulQuat(r_base2Tip, r_base_conj, r_tip);

        //apply kinematics to get joint angles
        //see Eq5 in Allen paper
        mjtNum R_base2Tip[9]{ 0 };
        mju_quat2Mat(R_base2Tip, r_base2Tip);
        mjtNum phi = mju_acos(R_base2Tip[8]);

        // avoid division by zero, see allen eq 14
        mjtNum u = 0;
        mjtNum v = 0;
        if (phi < 1e-6)
        {
            u = R_base2Tip[7];
            v = R_base2Tip[2];
        }
        else
        {
            u = R_base2Tip[7] * phi / mju_sin(phi);
            v = R_base2Tip[2] * phi / mju_sin(phi);
        }

        //set joint angles in sensordata
        mysensordata[0] = u;
        mysensordata[1] = v;

        //FOR JOINT VELOCITIES
        //get sensor id for frameangvel sensor on last disk of joint. Note that frameangvel is referenced to base disk in xml, not referenced to global coords.
        std::string tip_angvel_name = sensorName + "_B" + std::to_string(int(*numDisks) - 1) + "_frameangvel";
        int tip_angvel_id = mj_name2id(m, mjOBJ_SENSOR, tip_angvel_name.c_str());
        if (tip_angvel_id == -1)
        {
            mju_error("tip_angvel_id is -1");
            return;
        }

        //get angular velocity of first and last disk in joint frame
        int tip_angvel_adr = m->sensor_adr[tip_angvel_id];
        mjtNum wx = d->sensordata[tip_angvel_adr];
        mjtNum wy = d->sensordata[tip_angvel_adr + 1];

        // set joint velocities in sensordata
        //this part seems a little silly. I'm coping angvel from sensordata and putting it into another part of sensor data. Necessary?

        mysensordata[2] = wx;
        mysensordata[3] = wy;

        mj_freeStack(d);
    }

    void JointAngleEstimator::Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
        mjvScene* scn, int instance) {
        mj_markStack(d);

        // Get sensor id.
        int id;
        for (id = 0; id < m->nsensor; ++id) {
            if (m->sensor_type[id] == mjSENS_PLUGIN &&
                m->sensor_plugin[id] == instance) {
                break;
            }
        }

        // Get sensor data.
        mjtNum* sensordata = d->sensordata + m->sensor_adr[id];


        mj_freeStack(d);
    }


    void JointAngleEstimator::RegisterPlugin() {

        mjpPlugin plugin;
        mjp_defaultPlugin(&plugin);

        plugin.name = "mujoco.sensor.joint_angle_estimator";
        plugin.capabilityflags |= mjPLUGIN_SENSOR;

        // Parameterized by 4 attributes.
        const char* attributes[] = { "nchannel" };
        plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
        plugin.attributes = attributes;

        // Stateless.
        plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

        // Sensor dimension = nchannel
        plugin.nsensordata = +[](const mjModel* m, int instance, int sensor_id) {
            // u, v, udot, vdot
            return 4;
            };

        // Can only run after forces have been computed.
        plugin.needstage = mjSTAGE_ACC;

        // Initialization callback.
        plugin.init = +[](const mjModel* m, mjData* d, int instance) {
            auto* JointAngleEstimator = JointAngleEstimator::Create(m, d, instance);
            if (!JointAngleEstimator) {
                return -1;
            }
            d->plugin_data[instance] = reinterpret_cast<uintptr_t>(JointAngleEstimator);
            return 0;
            };

        // Destruction callback.
        plugin.destroy = +[](mjData* d, int instance) {
            delete reinterpret_cast<JointAngleEstimator*>(d->plugin_data[instance]);
            d->plugin_data[instance] = 0;
            };

        // Reset callback.
        plugin.reset = +[](const mjModel* m, double* plugin_state, void* plugin_data,
            int instance) {
                auto* JointAngleEstimator = reinterpret_cast<class JointAngleEstimator*>(plugin_data);
                JointAngleEstimator->Reset(m, instance);
            };

        // Compute callback.
        plugin.compute =
            +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
            auto* JointAngleEstimator =
                reinterpret_cast<class JointAngleEstimator*>(d->plugin_data[instance]);
            JointAngleEstimator->Compute(m, d, instance);
            };

        // Visualization callback.
        plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt,
            mjvScene* scn, int instance) {
                auto* JointAngleEstimator =
                    reinterpret_cast<class JointAngleEstimator*>(d->plugin_data[instance]);
                JointAngleEstimator->Visualize(m, d, opt, scn, instance);
            };

        // Register the plugin.
        mjp_registerPlugin(&plugin);
    }

}  // namespace mujoco::plugin::sensor