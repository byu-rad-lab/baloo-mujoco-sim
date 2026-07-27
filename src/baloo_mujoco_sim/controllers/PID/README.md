# PID Hugger

## 🚀 Getting Started

### Prerequisites
There might be some packages you need to install to run this script. 

Run evaluate_PID in terminal change the num_trials to adjust the numnber of trials that are run in the back ground and num-visual to run as many rollouts you want to see a mujoco window for:
python3 evaluate_pid_1000.py --num_trials 1000 --num_visual 10


### Installation & Usage
In order to run the PID Huggers script you downlaod them both and run the `view_openloop.py` script. 

view_openloop.py: The view_openloop.py script will be looking for a file called open_loop_hugger.py which is the same title open loop hugger from the baloo gym script.

open_loop_hugger.py: I copied this code under that file for convienence so that I could run simulations quickly.

Author's Note:
This is my first committ to github so appologies if the way to run this script ain't great

-Cameron Collyer June 2026

```bash
python view_openloop.py

For EvaluatePID_1000WithHistogramsAnd6PIDControllers

This script gets the maximum pressures of running a certain number of trials in simulation to run the amount of trials that you want I use the following bash command 

cameronc@salmon-vr:~$ /usr/bin/python /home/cameronc/baloo_ws/src/EvaluatePID_1000WithHistogramsAnd6PIDControllers.py --num_trials 1000 --num_detail 0

--num_trials specifies the number of headless trials you want to run while --num_detail specifies the number of trials you want to visualize. The num_detail trials usually crash on me quite frequently and it will say core dumped. If this happens just run it again. But if you want to reliably get data just run the script with headless trials. The script is also equipt to have an individual PID controller on each joint . Left J0,J1,J2 Right J0,J1,J2. Each of the gains have been calculated from running optuna with a fixed correction max based on the correction max chosen by the reinforcement learning policy. This script was used to run 1000 trials in simulation and verifies that the pressures applied in simulation do not go over the pressure threshold. The maximum pressures applied are exported to histograms. 

These are all the packages needed to run the script:


import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sys.path.append('/home/cameronc/baloo_ws/src/baloo-gym/src')

import baloo_gym.policies.PIDHugger as pid_hugger
pid_hugger.OpenLoopHuggerPolicy.print_pressures = lambda self, *args, **kwargs: None
pid_hugger.OpenLoopHuggerPolicy.save_logs = lambda self, *args, **kwargs: None

import mujoco

