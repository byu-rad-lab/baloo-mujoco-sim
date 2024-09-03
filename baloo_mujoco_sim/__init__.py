import importlib.metadata
import glob
import os


def find_xml_files_in_assets(package_name):
    try:
        # Get the distribution object for the package
        distribution = importlib.metadata.distribution(package_name)

        # Locate the base directory of the package
        package_location = distribution.locate_file('')

        # Construct the path to the 'assets' directory
        assets_path = os.path.join(package_location, package_name + '/assets')

        # print(f"Searching for XML files in '{assets_path}'...")

        # Search for XML files in the 'assets' directory
        xml_files = glob.glob(os.path.join(assets_path, '*.xml'))

        return xml_files

    except importlib.metadata.PackageNotFoundError:
        print(f"Package '{package_name}' not found.")
        return []


# Example usage
package_name = 'baloo_mujoco_sim'
xml_files = find_xml_files_in_assets(package_name)

if xml_files:
    XML_PATH = xml_files[0]
else:
    raise FileNotFoundError(
        f"No XML files found in '{package_name}' package. Remember to run generate-baloo-xml."
    )
