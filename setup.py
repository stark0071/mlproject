from typing import List
from setuptools import find_packages, setup


HYPEN_DASH_E_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    """Return list of requirements from requirements.txt"""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPEN_DASH_E_DOT in requirements:
            requirements.remove(HYPEN_DASH_E_DOT)
    return requirements



setup(
    name="mlproject",
    version="0.0.1",
    author="Prathamesh Pachade",
    author_email="pachade2@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)