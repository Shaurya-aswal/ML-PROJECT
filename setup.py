from setuptools import setup, find_packages   
from typing import List

def get_requirements(file_path: str) -> List[str]:
    
    '''Reads the requirements from a file and returns them as a list of strings.'''
    
    with open(file_path, "r") as file:
        requirements = file.readlines()
    return [req.replace("\n", "") for req in requirements]



setup(
    name="ml_project",
    version="0.1",
    author="Shaurya",
    author_email="shaurya.aswal12@gmail.com",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages(),
)          