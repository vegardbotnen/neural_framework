from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["networkx", "matplotlib", "numpy"]

setup(
    name="neural_framework",
    version="0.0.9",
    author="Volanpar",
    author_email="",
    description="Neural computation framework.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/volanpar/neural_framework",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)