"""
Setup script for the QuickMasker application.

This script uses setuptools to package the QuickMasker application,
allowing it to be installed and run as a command-line tool.
"""

from setuptools import setup, find_packages
from pathlib import Path

### Project Metadata
NAME = "QuickMasker"
VERSION = "0.1.0"
DESCRIPTION = "Simple SAM-based tool for interactive image segmentation and annotation."
AUTHOR = "Tom Olesch"
URL = "https://github.com/tolesch/QuickMasker"

# Read README for Long Description
try:
    HERE = Path(__file__).parent.resolve()
    LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION # Fallback if README.md is not found

### Dependencies
REQUIRED_PACKAGES = [
    "opencv-python",
    "numpy",
    "PyYAML",
    "pycocotools",
    "tqdm",
    "torch>=1.7",      
    "torchvision>=0.8",
    'segment-anything @ git+https://github.com/facebookresearch/segment-anything.git'
]

### Setup Configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    url=URL,
    project_urls={
        "Bug Tracker": f"{URL}/issues",
        "Source Code": URL,
    },
    packages=find_packages(where=".", include=['quickmasker', 'quickmasker.*']),
    
    # List of dependencies
    install_requires=REQUIRED_PACKAGES,

    # Define entry points for command-line scripts
    # creates a 'quickmasker' command that runs the main function
    # from quickmasker.run_quickmasker module.
    entry_points={
        "console_scripts": [
            "quickmasker=quickmasker.run_quickmasker:main",
        ],
    },


    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities",
    ],

    python_requires=">=3.8",
)

print(f"\nTo install {NAME} locally for development, run:")
print("  pip install -e .")
print(f"After installation, you can run the tool using the command: quickmasker")
