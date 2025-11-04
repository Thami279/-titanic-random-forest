"""Setup configuration for titanic_ml package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="titanic-ml",
    version="0.1.0",
    description="Production-ready ML package for Titanic survival prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scikit-learn==1.4.2",
        "matplotlib==3.8.4",
        "seaborn==0.13.2",
        "scipy==1.11.4",
        "fastapi==0.111.0",
        "uvicorn[standard]==0.29.0",
    ],
    extras_require={
        "mlflow": ["mlflow==2.16.2"],
        "dev": [
            "pytest==8.3.3",
            "pytest-cov==5.0.0",
            "black==24.8.0",
            "ruff==0.6.8",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
