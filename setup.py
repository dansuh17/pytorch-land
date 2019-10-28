import pathlib
from setuptools import setup, find_packages

ROOT_PATH = pathlib.Path(__file__).parent
README_TEXT = (ROOT_PATH / 'README.md').read_text()

setup(
    name='pytorch-land',
    version='0.1.3',
    description='pytorch-land for happy deep learning',
    long_description=README_TEXT,
    long_description_content_type='text/markdown',
    author='dansuh17',
    author_email='kaehops@gmail.com',
    url='https://github.com/dansuh17/pytorch-land',
    license='MIT',
    packages=find_packages(),
)
