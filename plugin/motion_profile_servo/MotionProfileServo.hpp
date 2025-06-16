#ifndef MUJOCO_PLUGIN_ACTUATOR_MOTIONPROFILESERVO_H_
#define MUJOCO_PLUGIN_ACTUATOR_MOTIONPROFILESERVO_H_

#include <memory>
#include <optional>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>

namespace mujoco::plugin::actuator {

  struct ServoConfig {
    double p_gain = 0.0;
    double v_gain = 0.0;
    double zeta = 0.0;
    double wn = 0.0;

    // Reads plugin attributes to construct PID configuration.
    static ServoConfig FromModel(const mjModel* m, int instance);
  };

  // An actuator plugin which implements configurable PID control.
  class MotionProfileServo {
  public:
    // Returns an instance of MotionProfileServo. The result can be null in case of
    // misconfiguration.
    static std::unique_ptr<MotionProfileServo> Create(const mjModel* m, int instance);

    // Returns the number of state variables for the plugin instance
    static int StateSize(const mjModel* m, int instance);

    // Resets the C++ MotionProfileServo instance's state.
    // plugin_state is a C array pointer into mjData->plugin_state, with a size
    // equal to the value returned from StateSize.
    void Reset(mjtNum* plugin_state);

    // Computes the rate of change for activation variables
    void ActDot(const mjModel* m, mjData* d, int instance) const;

    // Idempotent computation which updates d->actuator_force and the internal
    // state of the class. Called after ActDot.
    void Compute(const mjModel* m, mjData* d, int instance);

    // Updates plugin state.
    void Advance(const mjModel* m, mjData* d, int instance) const;

    // Adds the PID plugin to the global registry of MuJoCo plugins.
    static void RegisterPlugin();

  private:
    MotionProfileServo(ServoConfig config, std::vector<int> actuators);

    // Returns the number of activation variables for the plugin instance
    static int ActDim(const mjModel* m, int instance, int actuator_id);

    //? this plugin doesn't have a "state" as far as mujoco is concerned, I think this is just internal. Helpers.
    struct State {
      mjtNum prefiltered_position_command_dot = 0;
      mjtNum prefiltered_position_command = 0;
    };

    struct StateDot {
      mjtNum prefiltered_position_command_ddot = 0;
      mjtNum prefiltered_position_command_dot = 0;
    };

    // Reads data from d->act and returns it as a State struct.
    State GetPrefilterState(const mjModel* m, mjData* d, int actuator_idx) const;

    StateDot GetPrefilterStateDot(const mjModel* m, const mjData* d, int actuator_idx, mjtNum ctrl,
      const State& state) const;

    // Returns the PID setpoint, which is normally d->ctrl, but can be d->act for
    // actuators with dyntype != none.
    mjtNum GetCtrl(const mjModel* m, const mjData* d, int actuator_idx,
      const State& state, bool actearly) const;


    ServoConfig config_;
    // set of actuator IDs controlled by this plugin instance.
    std::vector<int> actuators_;
  };

}  // namespace mujoco::plugin::actuator

#endif  // MUJOCO_PLUGIN_ACTUATOR_MOTIONPROFILESERVO_H_