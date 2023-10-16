from setuptools import setup

setup(
   name='clinalyse',
   version='1.0.0',
   description='Module for analysing multi-locus clines',
   author='Nina Haladova & Stuart JE Baird',
   author_email='ninahaladova@gmail.com',
   packages=['clinalyse'],
   install_requires=['matplotlib', 'numpy', 'pandas', 'psutil', 'scipy'],
)

