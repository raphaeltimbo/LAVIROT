# from distutils.core import setup
from setuptools import setup

setup(name='LaviRot',
      version='0.1.4b',
      description='LaviRot',
      author='Lavi',
      author_email='raphaelts@gmail.com',
      packages=['LaviRot'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
     )