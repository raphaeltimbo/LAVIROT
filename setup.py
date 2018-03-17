# from distutils.core import setup
from setuptools import setup

setup(name='lavirot',
      version='0.1.5b',
      description='lavirot',
      author='Lavi',
      author_email='raphaelts@gmail.com',
      packages=['lavirot'],
      package_data={'lavirot': ['styles/*']},
      setup_requires=['pytest-runner'],
      tests_require=['pytest']
     )
