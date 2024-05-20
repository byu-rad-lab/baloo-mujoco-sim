from dm_control import mjcf
import mujoco
import mujoco.viewer as viewer

# load baloo xml file
mjcf_model = mjcf.from_path("/home/curtis/baloo_mujoco_sim/baloo.xml")

#save mjcf model to xml file
xml = mjcf_model.to_xml_string("/home/curtis/baloo_mujoco_sim/baloo2.xml")

# save xml
with open("baloo2.xml", "w") as f:
    f.write(xml)

# Parse from file.
# path = "/home/curtis/baloo_mujoco_sim/baloo.xml"
# model = mujoco.MjModel.from_xml_path(path)

##LOOP
# get mocap pose over time

#feed mocap pose into IK solver I wrote

# now I have desired joint angles, fed them into joint angle controller to get ctrl signal

#fill ctrl signals

#simulate
