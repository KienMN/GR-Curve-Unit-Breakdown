import os
from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))

readme = open(path + "/README.md")

setup(
  name="gr-curve-unit-breakdown",
  version="1.0.0",
  description="Several functions to work with GR curve.",
  url="https://github.com/KienMN/GR-Curve-Unit-Breakdown",
  author="Kien MN",
  author_email="kienmn97@gmail.com",
  license="MIT",
  packages=find_packages(exclude=["docs","tests", ".gitignore"]),
  install_requires=["numpy", "pandas", "matplotlib", "seaborn", "pywt", "statsmodels"],
  dependency_links=[""],
  include_package_data=True
)
