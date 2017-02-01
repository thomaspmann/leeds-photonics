from setuptools import setup, find_packages

setup(
    name='photonics',  # This is the name of your PyPI-package.
    version='0.1',    # Update the version number for new releases
    description='Extracting lifetimes from fluorescence measurements',
    author='Thomas Mann',
    author_email='mn14tm@leeds.ac.uk',
    license='MIT',
    url='http://github.com/mn14tm/photonics',
    download_url='https://github.com/mn14tm/photonics/tarball/0.1',
    keywords='fluorescence fluorescence photonics fluorescence',
    packages=['photonics'],
    install_requires=find_packages()
)
