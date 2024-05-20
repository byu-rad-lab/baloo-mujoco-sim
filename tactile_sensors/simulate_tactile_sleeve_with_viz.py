from time import time
import mujoco
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import mujoco.viewer

# path to robot description file
XML_PATH = "./tactile_sleeve.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# data for plotting post-simulation
tactile_sensor_data = np.ones((30, 30))

# time length of simulation
duration = 30.0  # (seconds)

# defining start time and desired time step for visualization to render
start = data.time

# Create the heatmap plot
fig, ax = plt.subplots()
heatmap = ax.imshow(tactile_sensor_data, cmap="hot")

# Set up the figure to update dynamically
plt.ion()  # Turn on interactive mode

# Display the initial heatmap
plt.show()

heatmap_hist = []
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # integrate continous dynamics
        mujoco.mj_step(model, data)

        viewer.sync()

        # Update the heatmap data directly
        tactile_signals = deepcopy(data.sensordata.astype(int).reshape(30, 30))
        heatmap_hist.append(tactile_signals)
i =0
for tactile_signals in heatmap_hist:
    print(i)
    i+=1
    heatmap.set_array(tactile_signals)

    # Update the color scale
    heatmap.set_clim(vmin=tactile_signals.min(), vmax=tactile_signals.max())

    # Redraw only the colorbar
    fig.canvas.draw_idle()

    # Pause for a short interval to allow for smooth animation
    plt.pause(0.0000001)


# Turn off interactive mode to keep the final plot displayed
plt.ioff()

# Show the final plot
plt.show()
