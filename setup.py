import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CoCoNet",
    version="2.0-alpha",
    author="Stefano Demarchi, Andrea Gimelli, Giacomo Rosato",
    author_email="stefano.demarchi@edu.unige.it",
    license='GNU General Public License with Commons Clause License Condition v1.0',
    description="Tool for the creation and the conversion of Neural Network models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeVerTools/CoCoNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language:: Python:: 3.9",
        "Development Status:: 3 - Alpha",
        "Topic:: Scientific/Engineering:: Artificial Intelligence",
        "Operating System:: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=['pynever', 'PyQt6']
)
