#!/usr/bin/env python
from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

setup(
    name='mmtafrica',
    version='0.0.1',
    description='MMTAfrica',
    author='Chris Emezue and Bonaventure Dossou',
    url='https://github.com/edaiofficial/mmtafrica',
    license='Apache License',
    install_requires=install_requires,
    packages=find_packages(exclude=[]),
    python_requires='>=3.7',
    project_urls={
        'Documentation': 'https://github.com/edaiofficial/mmtafrica',
        'Source': 'https://github.com/edaiofficial/mmtafrica',
        'Tracker': 'https://github.com/edaiofficial/mmtafrica/issues',
    },
    entry_points={
        'console_scripts': [
        ],
    }
)