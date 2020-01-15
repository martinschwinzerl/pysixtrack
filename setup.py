import setuptools
import sys

requirements = {"install": ["numpy", "scipy"]}

if sys.version_info < (3, 7):
    requirements["install"].append("dataclasses")

version = open("pysixtrack/__init__.py").readline().split('"')[1]

setuptools.setup(
    name="pysixtrack",
    version=version,
    description="6D Tracking Code",
    author="Riccardo De Maria",
    author_email="riccardo.de.maria@cern.ch",
    url="https://github.com/rdemaria/pysixtrack",
    packages=["pysixtrack", "pysixtrack.be_beamfields"],
    package_dir={"pysixtrack": "pysixtrack"},
    install_requires=requirements["install"],
)
