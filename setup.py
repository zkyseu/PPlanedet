import setuptools
import sys

import pplanedet

long_description = "A Tookit for lane detection based on PaddlePaddle"

with open("requirements.txt") as file:
    REQUIRED_PACKAGES = file.read()

setuptools.setup(
    name="pplanedet",
    version='0.0.2',
    author="kunyangzhou",
    author_email="zhoukunyangmcgill@163.com",
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/zkyseu/PPlanedet",
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=['cython', 'numpy'],
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT License',
    entry_points={'console_scripts': ['pplanedet=pplanedet.command:main', ]})
