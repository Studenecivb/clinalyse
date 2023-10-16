from setuptools import setup

setup(
   name='clinalyse',
   version='1.0.0',
   description='Module for analysing multi-locus clines',
   author='Nina Haladova & Stuart JE Baird',
   author_email='ninahaladova@gmail.com',
   packages=['clinalyse'],
   install_requires=['matplotlib', 'numpy', 'pandas', 'psutil', 'scipy'],
   download_url='https://github.com/Studenecivb/clinalyse/archive/refs/tags/v1.0.0.tar.gz',
)

