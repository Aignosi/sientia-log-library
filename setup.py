from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sientia_tracker',
    version='1.0.2',
    author=['Ítalo Azevedo', 'Pedro Bahia'],
    author_email=['italo@aignosi.com.br', 'pedro.bahia@aignosi.com.br'],
    description='Library for Aignosi Tracking API',
    packages=['sientia_tracker'],
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',


    ],
)
