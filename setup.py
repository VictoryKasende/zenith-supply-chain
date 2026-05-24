from setuptools import find_packages, setup

setup(
    name="zenith-supply-chain",
    version="1.0.0",
    description=(
        "Planification intelligente de la chaîne d'approvisionnement — "
        "Zenith Informatique & Bureautique"
    ),
    author="KASENDE NGELEKA Victoire",
    author_email="victorykasende@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "statsmodels>=0.14.0",
        "pulp>=2.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tabulate>=0.9.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "zenith-pipeline=scripts.run_pipeline:main",
        ],
    },
)
