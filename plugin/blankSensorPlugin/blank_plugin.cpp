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

#include "blank_plugin.hpp"

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

    // Creates a Blank instance if all config attributes are defined and
    // within their allowed bounds.
    Blank* Blank::Create(const mjModel* m, mjData* d, int instance) {
        if (CheckAttr(std::string(mj_getPluginConfig(m, instance, "nchannel")))) {
            // nchannel
            int nchannel = strtod(mj_getPluginConfig(m, instance, "nchannel"), nullptr);
            if (!nchannel) nchannel = 1;
            if (nchannel < 1 || nchannel > 6) {
                mju_error("nchannel must be between 1 and 6");
                return nullptr;
            }

            return new Blank(m, d, instance, nchannel);
        }
        else {
            mju_error("Invalid or missing parameters in blank sensor plugin");
            return nullptr;
        }
    }

    Blank::Blank(const mjModel* m, mjData* d, int instance, int nchannel) : nchannel_(nchannel)
    {
        // Make sure sensor is attached to a site.
        for (int i = 0; i < m->nsensor; ++i) {
            if (m->sensor_type[i] == mjSENS_PLUGIN && m->sensor_plugin[i] == instance) {
                if (m->sensor_objtype[i] != mjOBJ_SITE) {
                    mju_error("Touch Grid sensor must be attached to a site");
                }
            }
        }

    }

    void Blank::Reset(const mjModel* m, int instance) {}

    void Blank::Compute(const mjModel* m, mjData* d, int instance) {
        mj_markStack(d);

        // Get sensor id.
        int id;
        for (id = 0; id < m->nsensor; ++id) {
            if (m->sensor_type[id] == mjSENS_PLUGIN &&
                m->sensor_plugin[id] == instance) {
                break;
            }
        }

        // Clear sensordata and distance matrix.
        mjtNum* sensordata = d->sensordata + m->sensor_adr[id];
        mju_zero(sensordata, m->sensor_dim[id]);

        //compute sensor data and fill appropriate sensordata array
        for (int i = 0; i < nchannel_; i++)
        {
            sensordata[i] = i;
        }

        mj_freeStack(d);
    }

    // Thickness of taxel-visualization boxes relative to contact distance.
    static const mjtNum kRelativeThickness = 0.02;

    void Blank::Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
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


    void Blank::RegisterPlugin() {

        mjpPlugin plugin;
        mjp_defaultPlugin(&plugin);

        plugin.name = "mujoco.sensor.blank";
        plugin.capabilityflags |= mjPLUGIN_SENSOR;

        // Parameterized by 4 attributes.
        const char* attributes[] = { "nchannel", "size", "fov", "gamma" };
        plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
        plugin.attributes = attributes;

        // Stateless.
        plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

        // Sensor dimension = nchannel * size[0] * size[1]
        plugin.nsensordata = +[](const mjModel* m, int instance, int sensor_id) {
            int nchannel = strtod(mj_getPluginConfig(m, instance, "nchannel"), nullptr);
            return nchannel;
            };

        // Can only run after forces have been computed.
        plugin.needstage = mjSTAGE_ACC;

        // Initialization callback.
        plugin.init = +[](const mjModel* m, mjData* d, int instance) {
            auto* Blank = Blank::Create(m, d, instance);
            if (!Blank) {
                return -1;
            }
            d->plugin_data[instance] = reinterpret_cast<uintptr_t>(Blank);
            return 0;
            };

        // Destruction callback.
        plugin.destroy = +[](mjData* d, int instance) {
            delete reinterpret_cast<Blank*>(d->plugin_data[instance]);
            d->plugin_data[instance] = 0;
            };

        // Reset callback.
        plugin.reset = +[](const mjModel* m, double* plugin_state, void* plugin_data,
            int instance) {
                auto* Blank = reinterpret_cast<class Blank*>(plugin_data);
                Blank->Reset(m, instance);
            };

        // Compute callback.
        plugin.compute =
            +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
            auto* Blank =
                reinterpret_cast<class Blank*>(d->plugin_data[instance]);
            Blank->Compute(m, d, instance);
            };

        // Visualization callback.
        plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt,
            mjvScene* scn, int instance) {
                auto* Blank =
                    reinterpret_cast<class Blank*>(d->plugin_data[instance]);
                Blank->Visualize(m, d, opt, scn, instance);
            };

        // Register the plugin.
        mjp_registerPlugin(&plugin);
    }

}  // namespace mujoco::plugin::sensor