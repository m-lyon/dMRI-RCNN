#!/usr/bin/env python3
'''Use this to install module'''

import os
from setuptools import setup, find_namespace_packages

install_deps = [
    'tensorflow>=2.6.0',
    'numpy',
    'einops',
    'nibabel'
]

version = '0.1.4'
this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


try:
    import git
except ModuleNotFoundError:
    raise RuntimeError(
        'Please ensure gitpython is installed. Can be installed via "pip install gitpython"'
    )


def pull_first():
    '''Pull LFS objects'''
    cwd = os.getcwd()
    gitdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(gitdir)
    g = git.cmd.Git(gitdir)
    try:
        g.execute(['git', 'lfs', 'pull'])
    except git.exc.GitCommandError:
        raise RuntimeError('Please ensure git-lfs is installed.')
    os.chdir(cwd)


pull_first()
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
