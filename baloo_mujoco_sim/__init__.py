import importlib.resources as pkg_resources
from importlib.metadata import version
import baloo_mujoco_sim.assets as assets
import baloo_mujoco_sim


# Define a function to get the path to the baloo.xml file
def get_baloo_xml_path():
    ver = version(baloo_mujoco_sim.__name__)
    filename = f'baloo_v{ver}.xml'  # The name of the file
    # Use the path function to get a path to the resource
    try:
        with pkg_resources.path(assets, filename) as p:
            xml_path = str(p)
        return xml_path
    except FileNotFoundError:
        xml_path = ''  # Return an empty string if the file is not found
        raise


# Call the function to get the path to the current baloo.xml file
XML_PATH = get_baloo_xml_path()
