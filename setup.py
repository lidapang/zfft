from setuptools import setup, find_packages

cmdclass = {}
ext_modules = []

DESCRIPTION = "zfft"
NAME = "zfft"
AUTHOR = "Adam Luchies"
AUTHOR_EMAIL="adamluchies@gmail.com"
URL = "https://github.com/aluchies/zfft"
VERSION = "0.0.1"
LICENSE = "BSD 3-Clause"

setup(
      name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      download_url=URL,
      license=LICENSE,
      packages=find_packages(exclude=['tests']),
      cmdclass=cmdclass,
      ext_modules=ext_modules)