#TODO: would be nice to run this as part of the install process...

import subprocess
from pathlib import Path as path


def find_xml_files_in_assets(package_name):
    """
    Find all xml files in the assets folder of the package.
    :param package_name: name of the package
    :return: list of xml files
    """
    assets_path = path(__file__).parent / 'assets'
    xml_files = list(assets_path.glob('*.xml'))
    return [str(xml_file) for xml_file in xml_files]


# Example usage
package_name = 'baloo_mujoco_sim'
xml_files = find_xml_files_in_assets(package_name)

if xml_files:
    XML_PATH = xml_files[0]
else:
    #there is not xml file. So we need to generate it.
    print("No xml model found. Automatically generating xml file...")

    script = path(__file__).parent / "utils/generate_baloo_xml.py"

    res = subprocess.run(['python', script], capture_output=True, text=True)

    if res.returncode == 0:
        print(res.stdout)
        xml_files = find_xml_files_in_assets(package_name)
        XML_PATH = xml_files[0]
        print(f"Generated xml file: {XML_PATH}")
    else:
        print(res.stderr)
        raise Exception("Error while generating xml file")
