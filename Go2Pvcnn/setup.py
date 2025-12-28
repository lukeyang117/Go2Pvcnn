from setuptools import setup, find_packages

setup(
    name="Go2Pvcnn",
    version="0.1.0",
    description="Go2 quadruped locomotion with PVCNN-based perception using Isaac Lab",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "gymnasium>=0.29.0",
        # Isaac Lab and RSL-RL should be installed separately
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

