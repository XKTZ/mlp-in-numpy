from setuptools import setup, find_packages

setup(
    name="mlp-numpy",
    version="1.0",
    packages=find_packages("./src"),
    package_dir={'': 'src'})
