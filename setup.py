from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sientia_tracker',
    version='1.0.1',
    author='Ítalo Azevedo',
    author_email='italo@aignosi.com',
    description='Library for Aignosi Tracking API',
    packages=['sientia_tracker'],
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',


    ],
)
