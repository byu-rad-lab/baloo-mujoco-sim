#ifndef MUJOCO_PLUGIN_ACTUATOR_RUCKIGACTUATOR_H_
#define MUJOCO_PLUGIN_ACTUATOR_RUCKIGACTUATOR_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>

#include <memory>
#include <optional>
#include <vector>

#include <ruckig/ruckig.hpp>

namespace mujoco::plugin::actuator {

  struct RuckigConfig {
    double max_velocity = 0.0;
    double max_acceleration = 0.0;
    double max_jerk = 0.0;
    double dt = 0.0;
    double kp = 0.0;
    double kv = 0.0;
    double ka = 0.0;

    // Reads plugin attributes to construct PID configuration.
    static RuckigConfig FromModel(const mjModel* m, int instance);
  };

  // An actuator plugin which implements configurable PID control.
  class RuckigActuator {
    public:
    // Returns an instance of RuckigActuator. The result can be null in case of
    // misconfiguration.
    static std::unique_ptr<RuckigActuator> Create(const mjModel* m, int instance);

    // Returns the number of state variables for the plugin instance
    static int StateSize(const mjModel* m, int instance);

    // Resets the C++ RuckigActuator instance's state.
    // plugin_state is a C array pointer into mjData->plugin_state, with a size
    // equal to the value returned from StateSize.
    void Reset(mjtNum* plugin_state);

    // Computes the rate of change for activation variables
    void ActDot(const mjModel* m, mjData* d, int instance);

    // Idempotent computation which updates d->actuator_force and the internal
    // state of the class. Called after ActDot.
    void Compute(const mjModel* m, mjData* d, int instance);

    // Updates plugin state.
    void Advance(const mjModel* m, mjData* d, int instance);

    // Adds the PID plugin to the global registry of MuJoCo plugins.
    static void RegisterPlugin();

    private:
    RuckigActuator(RuckigConfig config, std::vector<int> actuators);

    // Returns the number of activation variables for the plugin instance
    static int ActDim(const mjModel* m, int instance, int actuator_id);

    // Returns the PID setpoint, which is normally d->ctrl, but can be d->act for
    // actuators with dyntype != none.
    mjtNum GetCtrl(
      const mjModel* m,
      const mjData* d,
      int actuator_idx,
      bool actearly) const;

    RuckigConfig config_;

    // set of actuator IDs controlled by this plugin instance.
    std::vector<int> actuators_;

    //ruckig trajectory planner
    ruckig::Ruckig<1> trajectory_planner_;
    ruckig::InputParameter<1> input_param_;
    ruckig::OutputParameter<1> output_param_;
    void ConfigureRuckig();

  };

}  // namespace mujoco::plugin::actuator

#endif  // MUJOCO_PLUGIN_ACTUATOR_RUCKIGACTUATOR_H_