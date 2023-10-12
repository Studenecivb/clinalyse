from setuptools import setup

setup(
   name='clinalyse',
   version='1.0.0',
   description='Module for analysing multi-locus clines',
   author='Stuart JE Baird & Nina Haladova',
   author_email='stuartj.e.baird@gmail.com',
   packages=['clinalyse'],
   install_requires=['matplotlib', 'numpy', 'pandas', 'psutil', 'scipy'],
)

