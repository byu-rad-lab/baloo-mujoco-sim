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

namespace mujoco::plugin::sensor
{

    namespace
    {

        // Checks that a plugin config attribute exists.
        bool CheckAttr(const std::string &input)
        {
            char *end;
            std::string value = input;
            value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
            strtod(value.c_str(), &end);
            return end == value.data() + value.size();
        }

        // Converts a string into a numeric vector
        template <typename T>
        void ReadVector(std::vector<T> &output, const std::string &input)
        {
            std::stringstream ss(input);
            std::string item;
            char delim = ' ';
            while (getline(ss, item, delim))
            {
                CheckAttr(item);
                output.push_back(strtod(item.c_str(), nullptr));
            }
        }

        // Returns the index of the first value in `a` that x is less than or n if no
        // such value exists. See: https://stackoverflow.com/a/39100135.
        int LowerBound(const mjtNum a[], int n, mjtNum x)
        {
            int l = 0;
            int h = n;
            while (l < h)
            {
                int mid = (l + h) / 2;
                if (x <= a[mid])
                {
                    h = mid;
                }
                else
                {
                    l = mid + 1;
                }
            }
            return l;
        }

        // Two dimensional histogram function.
        void Histogram2D(const mjtNum x_data[], const mjtNum y_data[],
                         const mjtNum weights[], int n_data, const mjtNum x_edges[],
                         int n_x_edges, const mjtNum y_edges[], int n_y_edges,
                         mjtNum *histogram, int *counts)
        {
            for (int i = 0; i < n_data; ++i)
            {
                mjtNum x = x_data[i];
                mjtNum y = y_data[i];
                int x_idx = LowerBound(x_edges, n_x_edges, x);
                if (x_idx == 0 || x_idx == n_x_edges)
                {
                    continue;
                }
                int y_idx = LowerBound(y_edges, n_y_edges, y);
                if (y_idx == 0 || y_idx == n_y_edges)
                {
                    continue;
                }
                int index = (y_idx - 1) * (n_x_edges - 1) + (x_idx - 1);
                histogram[index] += weights[i];
                if (counts)
                {
                    counts[index]++;
                }
            }
        }

        // Evenly spaced numbers over a specified interval.
        void LinSpace(mjtNum lower, mjtNum upper, int n, mjtNum array[])
        {
            mjtNum increment = n > 1 ? (upper - lower) / (n - 1) : 0;
            for (int i = 0; i < n; ++i)
            {
                *array = lower;
                ++array;
                lower += increment;
            }
        }

        // Parametrized linear/quintic interpolated nonlinearity.
        mjtNum Fovea(mjtNum x, mjtNum gamma)
        {
            // Quick return.
            if (!gamma)
                return x;

            // Foveal deformation.
            mjtNum g = mjMAX(0, mjMIN(1, gamma));
            return g * mju_pow(x, 5) + (1 - g) * x;
        }

        // Make bin edges.
        void BinEdges(mjtNum *x_edges, mjtNum *y_edges, int size[2], mjtNum fov[2],
                      mjtNum gamma)
        {
            // Make unit bin edges.
            LinSpace(-1, 1, size[0] + 1, x_edges);
            LinSpace(-1, 1, size[1] + 1, y_edges);

            // Apply foveal deformation.
            for (int i = 0; i < size[0] + 1; i++)
            {
                x_edges[i] = Fovea(x_edges[i], gamma);
            }
            for (int i = 0; i < size[1] + 1; i++)
            {
                y_edges[i] = Fovea(y_edges[i], gamma);
            }

            // Scale by field-of-view.
            mju_scl(x_edges, x_edges, fov[0] * mjPI / 180, size[0] + 1);
            mju_scl(y_edges, y_edges, fov[1] * mjPI / 180, size[1] + 1);
        }

        // Permute 3-vector from 0,1,2 to 2,0,1.
        static void xyz2zxy(mjtNum *x)
        {
            mjtNum z = x[2];
            x[2] = x[1];
            x[1] = x[0];
            x[0] = z;
        }

        // In the functions below transforming Cartesian <-> spherical:
        //  - The frame points down the z-axis, so a=e=0 corresponds to (0, 0, -r).
        //  - azimuth (a) corresponds to positive rotation around -y (towards +x).
        //  - elevation (e) corresponds to positive rotation around +x (towards +y).

        // Transform Cartesian (x,y,z) to spherical (azimuth, elevation, radius).
        void CartesianToSpherical(const mjtNum xyz[3], mjtNum aer[3])
        {
            mjtNum x = xyz[0], y = xyz[1], z = xyz[2];
            aer[0] = mju_atan2(x, -z);
            aer[1] = mju_atan2(y, mju_sqrt(x * x + z * z));
            aer[2] = mju_sqrt(x * x + z * z + y * y);
        }

        // Transform spherical (azimuth, elevation, radius) to Cartesian (x,y,z).
        void SphericalToCartesian(const mjtNum aer[3], mjtNum xyz[3])
        {
            mjtNum a = aer[0], e = aer[1], r = aer[2];
            xyz[0] = r * mju_cos(e) * mju_sin(a);
            xyz[1] = r * mju_sin(e);
            xyz[2] = -r * mju_cos(e) * mju_cos(a);
        }

    } // namespace

    // Creates a Blank instance if all config attributes are defined and
    // within their allowed bounds.
    Blank *Blank::Create(const mjModel *m, mjData *d,
                         int instance)
    {
        if (CheckAttr(std::string(mj_getPluginConfig(m, instance, "gamma"))) &&
            CheckAttr(std::string(mj_getPluginConfig(m, instance, "nchannel"))))
        {
            // nchannel
            int nchannel = strtod(mj_getPluginConfig(m, instance, "nchannel"), nullptr);
            if (!nchannel)
                nchannel = 1;
            if (nchannel < 1 || nchannel > 6)
            {
                mju_error("nchannel must be between 1 and 6");
                return nullptr;
            }

            // size
            std::vector<int> size;
            std::string size_str = std::string(mj_getPluginConfig(m, instance, "size"));
            ReadVector(size, size_str.c_str());
            if (size.size() != 2)
            {
                mju_error("Both horizontal and vertical resolutions must be specified");
                return nullptr;
            }
            if (size[0] <= 0 || size[1] <= 0)
            {
                mju_error("Horizontal and vertical resolutions must be positive");
                return nullptr;
            }

            // field of view
            std::vector<mjtNum> fov;
            std::string fov_str = std::string(mj_getPluginConfig(m, instance, "fov"));
            ReadVector(fov, fov_str.c_str());
            if (fov.size() != 2)
            {
                mju_error(
                    "Both horizontal and vertical fields of view must be specified");
                return nullptr;
            }
            if (fov[0] <= 0 || fov[0] > 180)
            {
                mju_error("`fov[0]` must be a float between (0, 180] degrees");
                return nullptr;
            }
            if (fov[1] <= 0 || fov[1] > 90)
            {
                mju_error("`fov[1]` must be a float between (0, 90] degrees");
                return nullptr;
            }

            // gamma
            mjtNum gamma = strtod(mj_getPluginConfig(m, instance, "gamma"), nullptr);
            if (gamma < 0 || gamma > 1)
            {
                mju_error("`gamma` must be a nonnegative float between [0, 1]");
                return nullptr;
            }

            return new Blank(m, d, instance);
        }
        else
        {
            mju_error("Invalid or missing parameters in blank sensor plugin");
            return nullptr;
        }
    }

    Blank::Blank(const mjModel *m, mjData *d, int instance) {}

    void Blank::Reset(const mjModel *m, int instance) {}

    void Blank::Compute(const mjModel *m, mjData *d, int instance) {}

    void Blank::Visualize(const mjModel *m, mjData *d, const mjvOption *opt,
                          mjvScene *scn, int instance)
    {
    }

    void Blank::RegisterPlugin()
    {
        mjpPlugin plugin;
        mjp_defaultPlugin(&plugin);

        plugin.name = "mujoco.sensor.blank";
        plugin.capabilityflags |= mjPLUGIN_SENSOR;

        // Parameterized by 4 attributes.
        const char *attributes[] = {"nchannel", "size", "fov", "gamma"};
        plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
        plugin.attributes = attributes;

        // Stateless.
        plugin.nstate = +[](const mjModel *m, int instance)
        { return 0; };

        // Sensor dimension = nchannel * size[0] * size[1]
        plugin.nsensordata = +[](const mjModel *m, int instance, int sensor_id)
        {
            int nchannel = strtod(mj_getPluginConfig(m, instance, "nchannel"), nullptr);
            if (!nchannel)
                nchannel = 1;
            std::vector<int> size;
            std::string size_str = std::string(mj_getPluginConfig(m, instance, "size"));
            ReadVector(size, size_str.c_str());
            return nchannel * size[0] * size[1];
        };

        // Can only run after forces have been computed.
        plugin.needstage = mjSTAGE_ACC;

        // Initialization callback.
        plugin.init = +[](const mjModel *m, mjData *d, int instance)
        {
            auto *Blank = Blank::Create(m, d, instance);
            if (!Blank)
            {
                return -1;
            }
            d->plugin_data[instance] = reinterpret_cast<uintptr_t>(Blank);
            return 0;
        };

        // Destruction callback.
        plugin.destroy = +[](mjData *d, int instance)
        {
            delete reinterpret_cast<Blank *>(d->plugin_data[instance]);
            d->plugin_data[instance] = 0;
        };

        // Reset callback.
        plugin.reset = +[](const mjModel *m, double *plugin_state, void *plugin_data,
                           int instance)
        {
            auto *Blank = reinterpret_cast<class Blank *>(plugin_data);
            Blank->Reset(m, instance);
        };

        // Compute callback.
        plugin.compute =
            +[](const mjModel *m, mjData *d, int instance, int capability_bit)
        {
            auto *Blank =
                reinterpret_cast<class Blank *>(d->plugin_data[instance]);
            Blank->Compute(m, d, instance);
        };

        // Visualization callback.
        plugin.visualize = +[](const mjModel *m, mjData *d, const mjvOption *opt,
                               mjvScene *scn, int instance)
        {
            auto *Blank =
                reinterpret_cast<class Blank *>(d->plugin_data[instance]);
            Blank->Visualize(m, d, opt, scn, instance);
        };

        // Register the plugin.
        mjp_registerPlugin(&plugin);
    }

} // namespace mujoco::plugin::sensor