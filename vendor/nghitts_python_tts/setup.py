"""
Setup script for Vietnamese TTS
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="vietnamese-tts",
    version="1.0.0",
    author="NGHI-TTS",
    description="Vietnamese Text-to-Speech with advanced text preprocessing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://nghitts.app",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Keep Python 3.8 (Jetson/Linux) on known compatible runtime versions.
        "piper-tts>=1.2.0,<1.3.0; python_version < '3.9'",
        "onnxruntime>=1.15.1,<1.17.0; python_version < '3.9'",
        "numpy>=1.23.5,<1.24.0; python_version < '3.9'",
        "piper-tts>=1.2.0; python_version >= '3.9'",
        "onnxruntime>=1.16.0; python_version >= '3.9'",
        "numpy>=1.24.0; python_version >= '3.9'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
        ],
    },
)
