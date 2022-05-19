#!/usr/bin/env python3
'''Use this to install module'''

import os
from setuptools import setup, find_namespace_packages

install_deps = [
    'tensorflow>=2.6.0',
    'numpy',
    'einops',
    'nibabel',
    'tqdm',
]

extra_req = {'sh': ['dipy']}
scripts = [
    'dmri_rcnn/bin/run_dmri_rcnn.py',
    'dmri_rcnn/bin/dmri_rcnn_download_all_weights.py',
    'dmri_rcnn/bin/dmri_sh_baseline.py',
]

version = '0.4.0'
this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='dmri-rcnn',
    version=version,
    description='Diffusion MRI Recurrent CNN for Angular Super-resolution.',
    author='Matthew Lyon',
    author_email='matthewlyon18@gmail.com',
    url='https://https://github.com/m-lyon/dMRI-RCNN',
    download_url=f'https://https://github.com/m-lyon/dMRI-RCNN/archive/v{version}.tar.gz',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    license='MIT License',
    packages=find_namespace_packages(),
    install_requires=install_deps,
    extras_require=extra_req,
    scripts=scripts,
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows :: Windows 10'
    ],
    include_package_data=True,
    keywords=['ai', 'cv', 'computer-vision', 'mri', 'dmri', 'super-resolution', 'cnn']
)
