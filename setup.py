from setuptools import setup, find_packages

setup(
    name='lifetime',  # This is the name of your PyPI-package.
    version='0.1',  # Update the version number for new releases
    description='Extracting lifetimes from fitting measurements',
    author='Thomas Mann',
    author_email='mn14tm@leeds.ac.uk',
    license='MIT',
    url='http://github.com/mn14tm/lifetime',
    download_url='https://github.com/mn14tm/lifetime/tarball/0.1',
    keywords='fluorescence fitting lifetime fitting',
    packages=['lifetime'],
    install_requires=['numpy', 'scipy']
)
