# from distutils.core import setup
from setuptools import setup

setup(name='LaviRot',
      version='1.0',
      description='LaviRot',
      author='GLavi',
      author_email='raphaelts@gmail.com',
      packages=['LaviRot'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
     )