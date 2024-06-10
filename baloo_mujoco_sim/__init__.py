import os
import importlib.resources as pkg_resources
import baloo_mujoco_sim.assets

# Initialize XML_STRING to None
XML_STRING = None


# Define a function to read the baloo.xml file
def read_baloo_xml():
    global XML_STRING
    # Try to read the baloo.xml file
    try:
        with pkg_resources.open_text(baloo_mujoco_sim.assets,
                                     'baloo.xml') as f:
            XML_STRING = f.read()
    except FileNotFoundError:
        # Handle the error if the file is not found
        XML_STRING = "Error: baloo.xml file not found."
    except Exception as e:
        # Handle any other exceptions
        XML_STRING = f"Error reading baloo.xml: {e}"


# Call the function to read the current baloo.xml file
read_baloo_xml()
