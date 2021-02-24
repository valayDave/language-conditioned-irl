import os
from typing import List

import setuptools


def get_long_description() -> str:
    with open('README.md') as fh:
        return fh.read()


def get_required() -> List[str]:
    with open('requirements.txt') as fh:
        return fh.read().splitlines()


def get_version():
    with open(os.path.join('language_conditioned_rl', '__init__.py')) as fh:
        for line in fh:
            if line.startswith('__version__ = '):
                return line.split()[-1].strip().strip("'")


setuptools.setup(
    name='language_conditioned_rl',
    packages=setuptools.find_packages(),
    version='0.0.1',
    license='MIT',
    description='Language Conditioned RL Repo',
    author='Valay Dave',
    include_package_data=True,
    author_email='valaygaurang@gmail.com',
    url='https://github.com/valayDave/language-conditioned-irl',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    keywords=[],
    install_requires=get_required(),
    python_requires='>=3.6',
    py_modules=['language_conditioned_rl', ],
)
