import os

from setuptools import setup, find_packages

_HERE = os.path.abspath(os.path.dirname(__file__))
_README = os.path.join(_HERE, 'README.md')

if os.path.exists(_README):
    with open(_README, 'r', encoding='utf-8') as fh:
        long_description = fh.read()
else:                      # keep `pip install .` working from a stripped copy
    long_description = 'ISO-UNet: region-aware U-Net for precipitation d18O emulation.'

setup(
    name='iso_unet',
    version='0.1.0',
    description='ISO-UNet',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.10',
    keywords=['Deep Learning', 'Water Isotope', 'Climate Emulator'],
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=[
        'colorama',
        'tqdm',
        'numpy',
        'torch',
        'lightning',
        'scikit-learn',
    ],
)
