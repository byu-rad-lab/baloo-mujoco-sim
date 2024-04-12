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

#ifndef MUJOCO_PLUGIN_SENSOR_TOUCH_GRID_H_
#define MUJOCO_PLUGIN_SENSOR_TOUCH_GRID_H_

#include <optional>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>

namespace mujoco::plugin::sensor {
    class JointAngleEstimator {
    public:
        static JointAngleEstimator* Create(const mjModel* m, mjData* d, int instance);
        JointAngleEstimator(JointAngleEstimator&&) = default;
        ~JointAngleEstimator() = default;

        void Reset(const mjModel* m, int instance);
        void Compute(const mjModel* m, mjData* d, int instance);
        void Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
            mjvScene* scn, int instance);

        static void RegisterPlugin();

        int nchannel_;           // number of channels (1-6)


    private:
        JointAngleEstimator(const mjModel* m, mjData* d, int instance, int nchannel);
        std::vector<mjtNum> distance_;
    };

}  // namespace mujoco::plugin::sensor

#endif  // MUJOCO_PLUGIN_SENSOR_TOUCH_GRID_H_