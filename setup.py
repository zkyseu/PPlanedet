import setuptools
import sys

long_description = "A Tookit for lane detection based on PaddlePaddle"

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
    install_requires=[
        'ftfy', 'regex', 'paddleseg'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={'console_scripts': ['pplanedet=pplanedet.command:main', ]})