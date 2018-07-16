# from distutils.core import setup
from setuptools import setup

setup(name='ross',
      version='0.1.5b',
      description='ross',
      author='Lavi',
      author_email='raphaelts@gmail.com',
      packages=['ross'],
      package_data={'ross': ['styles/*']},
      setup_requires=['pytest-runner'],
      tests_require=['pytest']
     )
