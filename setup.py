"""Setup script for CS-DIP package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cs-dip",
    version="1.0.0",
    author="CS-DIP Authors",
    description="Curvature-Steered Deep Image Prior: A Geometric Architecture "
                "for Self-Supervised Inverse Problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cs-dip/cs-dip",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
