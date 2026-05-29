from setuptools import setup
import os
from glob import glob

package_name = "skyshade"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SkyShade Team",
    maintainer_email="your-team@uts.edu.au",
    description="SkyShade autonomous drone umbrella simulation nodes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node   = skyshade.perception_node:main",
            "flight_node       = skyshade.flight_node:main",
            "env_decision_node = skyshade.env_decision_node:main",
            "nav_safety_node   = skyshade.nav_safety_node:main",
        ],
    },
)
