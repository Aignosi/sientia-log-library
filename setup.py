from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sientia_log',
    version='1.0.0',
    author='Ítalo Azevedo',
    author_email='italo@aignosi.com',
    description='Library for Aignosi Log API',
    packages=['sientia_log'],
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',


    ],
)
