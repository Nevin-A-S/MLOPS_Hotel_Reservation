import setuptools
import setuptools.installer

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="hotel_reservation MLOps",
    version="0.0.1",
    author="Nevin A S",
    author_email="nevinajithkumar@gmail.com",
    description="A MLOps project for hotel reservation prediction",
    packages=setuptools.find_packages(),
    install_requires=requirements
    )
