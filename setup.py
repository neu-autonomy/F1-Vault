from setuptools import setup, find_packages

setup(
    name="f1-vault",
    version="0.1.0",
    author="Your Name",
    description="Physics-informed learning for robot-terrain interaction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "h5py>=3.8.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
    ],
)