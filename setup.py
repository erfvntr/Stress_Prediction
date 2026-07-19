from setuptools import find_packages, setup
from typing import List


def requirements(path: str)-> List[str]:
    requirements = []
    with open(path) as file:
        reqs = file.readlines()
        reqs = [req.replace("\n", "") for req in reqs]
        
        if "-e ." in requirements:
            requirements.remove("-e .")
            
    return requirements
            
setup(
    name = "Stress Prediction",
    version = "0.0.1",
    author = "erfvntr",
    packages=find_packages(),
    install_requires=requirements("requirements.txt")
)