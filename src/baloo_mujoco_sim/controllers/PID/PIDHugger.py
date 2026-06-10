import numpy as np # Creates a claass object of numpy called np
from baloo_gym.utils.observation_spaces import StateObservationObjectOnly # Imports specifically the StateObservationObjectOnly class wich contains the varriables for anything to do with Baloo
from stable_baselines3.common.policies import BasePolicy # Inherets the BasePolicy from stable baselines.


# =============================================================================
# Slip Correction PID
# =============================================================================
# Monitors box z-position during LIFT. If the box drops (slipping), it outputs
# a positive squeeze_correction value which is added to the j1 antagonist
# pressures on both arms to tighten the grip.
#
# Tuning guide:
#   kp  — how aggressively to respond to a drop. Start low (50-100).
#          Too high → arms slam shut and vibrate.
#   ki  — slowly builds squeeze for sustained slip. Keep very small (1-5).
#          Too high → integral windup clamps arms permanently.
#   kd  — damps oscillation. Usually 1-3.
#   threshold — deadband in meters. Corrections only fire outside this range,
#               which prevents over-squeezing on small sensor noise.
#   correction_max — hard cap on how much extra delta-pressure the PID
#               can add. 50 PSI is a conservative starting point given the
#               hardware range of ±150 PSI delta.
# =============================================================================

class SlipCorrectionPID:

    def __init__( #This function sets up data so taht it is accessable to other functions
        self,
        kp=80.0, #Setting up the weights for the proportional derivative and integral terms.
        ki=2.0,
        kd=1.5,
        dt=0.01,
        threshold=0.01, # This is the threshold of movement the system tolerates before the pid activates I think measured in meters so it triggers at 1 cm of movement
        correction_min=0.0,   # Applies 0 additional KPA as the floor it can't take away pressure.
        correction_max=50.0,  # Maximum of 50 kpa applied with PID
    ):
        self.kp = kp # These lines copy data into self so that the rest of the class can access it.
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.threshold = threshold
        self.correction_min = correction_min
        self.correction_max = correction_max

        self._integral = 0.0 # This is a varrible that we need to accumulate the erorr for the integral term so that it can adjust. It keeps tract of the total error.
        self._prev_error = 0.0 # This keeps track of the previous error so that we can find the rate of change of error to help the derivative term do it's damping
        self.active = False # This is the initial state of the PID controller becaues we don't want it running until it grasps and experiences perturbations

    def arm(self, current_box_z): # This arms the PID controller
        self.desired_z = current_box_z
        self._integral = 0.0 # Just in case the arm function is run twice the error is reset to reset the derivative and integral terms
        self._prev_error = 0.0
        self.active = True # This is the line that activates it

    def update(self, box_z):
        if not self.active: # Does nothing if self.active == false
            return 0.0

        error = self.desired_z - box_z  # positive when box drops

        if abs(error) < self.threshold: #if the error is less than the threshold don't run PID because its just to small of a movement to matter
            return 0.0
        self._integral += error * self.dt # An approximation of a rhiemansum (computational integration)
        derivative = (error - self._prev_error) / self.dt # Finds the slope which is the derivative
        self._prev_error = error #Sets the current error to the previous error for the next iteration

        u = (self.kp * error
             + self.ki * self._integral
             + self.kd * derivative)  #This is the actual pid equation. The sum of the 3 parts

        return float(np.clip(u, self.correction_min, self.correction_max)) # numpy.clip (np.clip same thing) makes sure that if the value is above the max it will return the max or below the min that it will return the min
    
    def disarm(self): # Does the opposite of arm.
        self.active = False
        self._integral = 0.0
        self._prev_error = 0.0
        
# =============================================================================
# Squeeze direction reference
# =============================================================================
# Each arm has 3 joints, each with 2 DOFs (z-rotation, x-rotation).
# Actions are delta pressures from avg_pressure=150.
#
# Left arm:
#   actions[1] = left j0 DOF-z agonist   → shoulder bend DOF-z
#   actions[2] = left j0 DOF-z antagonist
#   actions[3] = left j1 DOF-z agonist   ← PRIMARY SQUEEZE (inward curl)
#   actions[4] = left j1 DOF-z antagonist ← PRIMARY SQUEEZE
#   actions[5] = left j2 DOF-z agonist   (unused/wrist)
#   actions[6] = left j2 DOF-z antagonist (unused/wrist)
#
# Right arm (mirror):
#   actions[7]  = right j0 DOF-z agonist
#   actions[8]  = right j0 DOF-z antagonist
#   actions[9]  = right j1 DOF-z agonist  ← PRIMARY SQUEEZE
#   actions[10] = right j1 DOF-z antagonist ← PRIMARY SQUEEZE
#   actions[11] = right j2 DOF-z agonist  (unused/wrist)
#   actions[12] = right j2 DOF-z antagonist (unused/wrist)
#
# At end of GRASP the j1 pressures are:
#   Left:  [actions[3], actions[4]] = [-100, +150]  → fully curled inward
#   Right: [actions[9], actions[10]] = [+45, -105]  → mirror curl
#
# squeeze_direction_left  = [-1, +1]  (decrease agonist, increase antagonist)
# squeeze_direction_right = [-1, +1]  (same pattern — decrease [9], increase [10] sign)
#
# The correction is ADDITIVE:
#   new_action[3]  = current[3]  + squeeze_correction * squeeze_dir_left[0]
#   new_action[4]  = current[4]  + squeeze_correction * squeeze_dir_left[1]
#   new_action[9]  = current[9]  + squeeze_correction * squeeze_dir_right[0]
#   new_action[10] = current[10] + squeeze_correction * squeeze_dir_right[1]
# =============================================================================

# Direction vectors: multiply correction by these to tighten each arm
SQUEEZE_DIR_LEFT  = np.array([-1.0, +1.0])   
SQUEEZE_DIR_RIGHT = np.array([-1.0, +1.0])    
SQUEEZE_DIR_LEFT_J2  = np.array([-1.0, +1.0])   # actions[5], actions[6]
SQUEEZE_DIR_RIGHT_J2 = np.array([1.0, -1.0])   # This helps j2 on the right arm to curl inward(cc arround the z axis if z is the direction baloo's elevator goes)actions[11], actions[12]
CHAMBER_MAP = {  #This is to map out the pressures to see what is going on inside baloo at real time
    "left_j0_agonist":    1,
    "left_j0_antagonist": 2,
    "left_j1_agonist":    3,
    "left_j1_antagonist": 4,
    "left_j2_agonist":    5,
    "left_j2_antagonist": 6,
    "right_j0_agonist":   7,
    "right_j0_antagonist":8,
    "right_j1_agonist":   9,
    "right_j1_antagonist":10,
    "right_j2_agonist":   11,
    "right_j2_antagonist":12,
}

class OpenLoopHuggerPolicy(BasePolicy): 

    def __init__(self, N, slip_pid_params=None):
        super().__init__(observation_space=None, action_space=None)

        # ── Time-Delay Gate for Slip Monitoring ─────────────────────────────
        self.lift_time = 0.0
        self.min_height = -.9 # The lowest height baloo's elevator can go to is down .9 meters
        self.step_along_trajectory = 0 # Initializes Baloo to be at the top of the elevator. This varriable will be used to control how Baloo moves down the elevator in later steps
        self.state = "APPROACH" # Created 3 different states to teach baloo what do before he has a box, when its time to grasp a box and when it's time to lift. This helps to differentiate arm movements to what we want them to be at these states.
        self.prev_actions = np.zeros(13) #Creates a vector of 13 zeros there are 6 pressure chambers used in each arm. and the last value is for Baloo's elevator posistion

        self.N = N # N is for the number of time steps
        self.actions_lb = np.asarray([-900] + [-150] * 12) # sets up an array so that the starting values of all the pressure chambers are 150kpa like Curtis talks about in the paper
        self.actions_ub = np.asarray([0]    + [150]  * 12)

        self.avg_pressure = 150 # This is the starting pressure that Curtis outlines in his paper then Baloo shifts his geometry by increasing or decreasing the pressure in joints
        
        # The next couple of lines uses the values Curtis calculated for the oppen loop hugger. They contain cordinates of where the joints begin and end then each line creates
        # an array that subtracts the first value from the last then divides that by the number of time steps and sends each coordinate to Baloo for smooth movement. Example
        # N =6 we're trying to go from 0 to 85. 85-0 = 85  [0,17,34,51,68,85]
        self.left_lift_j0_delta_traj = np.linspace(
            np.array([0, 0]), np.array([85, 0]), N)
        self.left_grab_j0_delta_traj = np.linspace(
            np.array([85, 0]), np.array([0, 150]), N)

        self.left_lift_j1_delta_traj = np.linspace(
            np.array([0, 0]), np.array([45, 45]), N)
        self.left_grab_j1_delta_traj = np.linspace(
            np.array([45, 45]), np.array([-100, 150]), N)

        self.right_lift_j0_delta_traj = np.linspace(
            np.array([0, 0]), np.array([-20, 80]), N)
        self.right_grab_j0_delta_traj = np.linspace(
            np.array([-20, 80]), np.array([150, -50]), N)

        self.right_lift_j1_delta_traj = np.linspace(
            np.array([0, 0]), np.array([50, 50]), N)
        self.right_grab_j1_delta_traj = np.linspace(
            np.array([50, 50]), np.array([45, -105]), N)

        # ── Slip correction PID ─────────────────────────────────────────────
        pid_params = slip_pid_params or {}
        self.slip_pid = SlipCorrectionPID(**pid_params) #Creates a class object with the PID dictionary being unpacked into the Slip Correction PID function
        self.slip_pid_j2 = SlipCorrectionPID(kp=100.0, ki=0, kd=0) #1,1

    # _predict method is required when you inheret from BasePolicy This function does nothing but makes sure that python functions
    def _predict(self, observation, deterministic=False):
        return None

    def normalize_actions(self, commands): #The RL policy expects values between [-1 and 1] This normalizes all inputs to a number within that range
        return 2 * (commands - self.actions_lb) / (self.actions_ub - self.actions_lb) - 1
    # This is where the meat of the PID lives 
    def predict(self, obs, deterministic=True,exact_box_z=None):
        actions = self.prev_actions.copy() #Takes a copy of the previous actions

        if len(obs) > 1: # Checks if obs is a single value or if it is a vector. If it's a vector
            mujoco_observation = StateObservationObjectOnly.from_standardized_array(obs) # Saves the values of the array into an object. Now its more clear what data
            #is in each value in the obs array it allows the next line to exist and clearly state what is going on
            elevator_height = mujoco_observation.elevator_pos #Copies the Observed elevator position into a varriable
        else: # If its a single value
            elevator_height = obs # The only value in the observation is a position for the elevator to go to

        # ── Parse observation ─────────────────────────────────────────────
        if exact_box_z is not None: #Self explanitory if the box height doesn't == nothing copy it to box_z 
            box_z = exact_box_z
        elif len(obs) > 1:
            box_z = mujoco_observation.chest2box[2] # Gets the z distance from th chest to the box centroid. This is one of the things the PID is targeting
        else:
            box_z = None  #There ain't a box here 
        # ── APPROACH ─────────────────────────────────────────────────────
        if self.state == "APPROACH":
            self.slip_pid.disarm() #PID is not active during the approach phase

            if self.step_along_trajectory < self.N:
                actions[0] = -900 #Move the elevator to the bottom
                #These actions run the open loop hugger
                actions[1] = self.left_lift_j0_delta_traj[self.step_along_trajectory][0]
                actions[2] = self.left_lift_j0_delta_traj[self.step_along_trajectory][1]
                actions[3] = self.left_lift_j1_delta_traj[self.step_along_trajectory][0]
                actions[4] = self.left_lift_j1_delta_traj[self.step_along_trajectory][1]
                actions[5] = 0
                actions[6] = 0

                actions[7]  = self.right_lift_j0_delta_traj[self.step_along_trajectory][0]
                actions[8]  = self.right_lift_j0_delta_traj[self.step_along_trajectory][1]
                actions[9]  = self.right_lift_j1_delta_traj[self.step_along_trajectory][0]
                actions[10] = self.right_lift_j1_delta_traj[self.step_along_trajectory][1]
                actions[11] = 0
                actions[12] = 0

                self.prev_actions = actions.copy()
                self.step_along_trajectory += 1

            # When the elevator gets clsoe to the bottom and we're at the
            #last time step in the trajectory trigger the grasp        
            if (np.isclose(elevator_height, -.900, atol=.1) and self.step_along_trajectory == self.N):
                self.state = "GRASP"
                self.step_along_trajectory = 0

        # ── GRASP ─────────────────────────────────────────────────────────
        #These are the inputs of the open loop hugger to grasp.
        elif self.state == "GRASP":
            actions[0] = -900

            actions[1] = self.left_grab_j0_delta_traj[self.step_along_trajectory][0]
            actions[2] = self.left_grab_j0_delta_traj[self.step_along_trajectory][1]
            actions[3] = self.left_grab_j1_delta_traj[self.step_along_trajectory][0]
            actions[4] = self.left_grab_j1_delta_traj[self.step_along_trajectory][1]
            actions[5] = 0
            actions[6] = 0

            actions[7]  = self.right_grab_j0_delta_traj[self.step_along_trajectory][0]
            actions[8]  = self.right_grab_j0_delta_traj[self.step_along_trajectory][1]
            actions[9] = self.right_grab_j1_delta_traj[self.step_along_trajectory][0]
            actions[10] = self.right_grab_j1_delta_traj[self.step_along_trajectory][1]
            actions[11] = 0
            actions[12] = 0
            
            self.prev_actions = actions.copy()
            self.step_along_trajectory += 1 # iterates the timestep
            #Similar trigger to last time. Once the last time step finishes switch to LIFT
            if self.step_along_trajectory == self.N:
                self.state = "LIFT"
                self.step_along_trajectory = 0
                # Arm the PID at the box height when we start lifting
                if box_z is not None:
                    self.slip_pid.arm(box_z) #Activates the PID
                    self.slip_pid_j2.arm(box_z)

        # ── LIFT ──────────────────────────────────────────────────────────
        elif self.state == "LIFT":
            # Tell the elevator to move towards the top.
            actions[0] = 0

            # Increment the time spent lifting (dt matches your PID configuration)
            self.lift_time += self.slip_pid.dt
            acceleration_window = 0.35 # The time we give the box to suddenly jolt before pid checks the relative velocty.
            # ── Slip correction ───────────────────────────────────────────
            if box_z is not None and self.lift_time >= acceleration_window:
                squeeze = self.slip_pid.update(box_z) #Saves an update on the box_z position to see if the 
                #position has changed or if the box is snugly in Baloo's loving arms
                squeeze_j2 = self.slip_pid_j2.update(box_z)
            else:
                squeeze = 0.0
                squeeze_j2 =0.0

            if squeeze > 0: #Box is dropping down out of the arms
                # Squeeze harder at j1
                actions[3]  = np.clip(
                    actions[3]  + squeeze * SQUEEZE_DIR_LEFT[0],  -150, 150)
                actions[4]  = np.clip(
                    actions[4]  + squeeze * SQUEEZE_DIR_LEFT[1],  -150, 150)
                actions[9]  = np.clip(
                    actions[9]  + squeeze * SQUEEZE_DIR_RIGHT[0], -150, 150)
                actions[10] = np.clip(
                    actions[10] + squeeze * SQUEEZE_DIR_RIGHT[1], -150, 150)
            squeeze_j2 = self.slip_pid_j2.update(box_z) if box_z is not None else 0.0

            if squeeze_j2 > 0:
                actions[5]  = np.clip(
                    actions[5]  + squeeze_j2 * SQUEEZE_DIR_LEFT_J2[0],  -150, 150)
                actions[6]  = np.clip(
                    actions[6]  + squeeze_j2 * SQUEEZE_DIR_LEFT_J2[1],  -150, 150)
                actions[11] = np.clip(
                    actions[11] + squeeze_j2 * SQUEEZE_DIR_RIGHT_J2[0], -150, 150)
                actions[12] = np.clip(
                    actions[12] + squeeze_j2 * SQUEEZE_DIR_RIGHT_J2[1], -150, 150)
                            
                
            #Uncomment these lines if you want to debug and see the pressure values
            #for name, idx in CHAMBER_MAP.items(): #Using the chamber map dictionary to evaluate the pressure values at all joints
            #   absolute_kpa = (self.avg_pressure + actions[idx])
            #    print(f"{name}: {float(absolute_kpa):.2f} kPa")
        norm_actions = self.normalize_actions(actions)
        return norm_actions, None

    # -------------------------------------------------------------------------
    def restart(self):
        self.step_along_trajectory = 0
        self.state = "APPROACH"
        self.prev_actions = np.zeros(13)
        self.slip_pid.disarm()
        self.slip_pid_j2.disarm()
        self.lift_time = 0.0  # Reset lift clock on overall restart