#!/usr/bin/env python3
'''Use this to install module'''

from os import path
from setuptools import setup, find_namespace_packages

install_deps = [
    'tensorflow>=2.6.0',
    'numpy',
    'einops',
    'nibabel'
]

version = '0.1.0'
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dmri-rcnn',
    version=version,
    description='Diffusion MRI Recurrent CNN for Angular Super-resolution.',
    author='Matt Lyon',
    author_email='matthewlyon18@gmail.com',
    url='https://https://github.com/mattlyon93/dMRI-RCNN',
    download_url='https://https://github.com/mattlyon93/dMRI-RCNN/archive/v{}.tar.gz'.format(version),
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    license='MIT License',
    packages=find_namespace_packages(),
    install_requires=install_deps,
    scripts=['dmri_rcnn/bin/run_dmri_rcnn.py'],
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows :: Windows 10'
    ],
    include_package_data=True,
    keywords=['ai', 'cv', 'computer-vision', 'mri', 'dmri', 'super-resolution', 'cnn']
)
