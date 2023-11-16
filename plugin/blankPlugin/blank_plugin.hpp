#ifndef MUJOCO_PLUGIN_SENSOR_BLANK_H_
#define MUJOCO_PLUGIN_SENSOR_BLANK_H_

#include <optional>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>

namespace mujoco::plugin::sensor
{
    class Blank
    {
    public:
        static Blank *Create(const mjModel *m, mjData *d, int instance);
        // move constructor
        Blank(Blank &&) = default;
        // Destructor
        ~Blank() = default;

        void Reset(const mjModel *m, int instance);
        void Compute(const mjModel *m, mjData *d, int instance);

        void Visualize(
            const mjModel *m,
            mjData *d,
            const mjvOption *opt,
            mjvScene *scn,
            int instance);

        static void RegisterPlugin();

    private:
        Blank(
            const mjModel *m,
            mjData *d,
            int instance);

        std::vector<mjtNum> distance_;
    };

} // namespace mujoco::plugin::sensor

#endif // MUJOCO_PLUGIN_SENSOR_BLANK_H_