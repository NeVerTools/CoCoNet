import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CoCoNet",
    version="1.0-beta",
    author="Dario Guidotti, Stefano Demarchi",
    author_email="{dario.guidotti}{stefano.demarchi}@edu.unige.it",
    license='GNU General Public License with Commons Clause License Condition v1.0',
    description="Tool for the creation and the conversion of Neural Network models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeVerTools/CoCoNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language:: Python:: 3.8",
        "Development Status:: 3 - Alpha",
        "Topic:: Scientific/Engineering:: Artificial Intelligence",
        "Operating System:: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['pynever', 'numpy', 'PyQt5', 'onnx', 'torch', 'torchvision', 'pysmt']
)
