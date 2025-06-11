#!/usr/bin/env python
# filepath: /home/emilio/Documents/ai/md/setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="proteinMD",
    version="0.1.0",
    author="Emilio",
    author_email="your.email@example.com",
    description="A comprehensive molecular dynamics simulation system for protein behavior in cellular environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/proteinMD",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "scipy>=1.8.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.1.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "analysis": [
            "mdtraj>=1.9.7",
            "seaborn>=0.11.2",
            "pandas>=1.4.0",
        ],
    },
)
