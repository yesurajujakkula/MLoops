# this setup.py will help create machine learning as whole as package
# so that we can send this package into the python pypy 
# we can use it any where

from setuptools import find_packages,setup
from typing import List
import os

def get_requirements(file_path:str) -> List[str]:
        """
        the given function will read the all the libraries present in the requirements.txt
        input: requirements.txt path
        output : list of libraries
        """
        full_file_path = os.path.join(os.getcwd(),file_path)
        requirements = []
        with open(full_file_path,'r') as f:
               requirements = f.readlines()
               # readlines function will return ln also that why we are removing it
               requirements = [lib.replace("\n","") for lib in requirements if "-e ." not in lib]
        return requirements

setup(
        name = "MLOOPs",
        version = "0.0.1",
        authour = "Yesu Raju",
        authour_email = "yesurajujakkula@gmail.com",
        packages=find_packages(),
        install_requires = get_requirements("requirements.txt")
)