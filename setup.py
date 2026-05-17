from setuptools import setup, find_packages

setup(
    name="zenith-supply-chain",
    version="1.0.0",
    description="Planification intelligente de la chaîne d'approvisionnement — Zenith Informatique",
    author="KASENDE NGELEKA Victoire",
    author_email="victorykasende@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "tensorflow>=2.13.0",
        "statsmodels>=0.14.0",
        "pmdarima>=2.0.0",
        "pulp>=2.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)
