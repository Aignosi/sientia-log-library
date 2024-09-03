from setuptools import setup, find_packages


with open("README.md", "r") as arq:
    readme = arq.read()

# with open("requirements.txt", "r") as req_file:
#     requirements = req_file.read().splitlines()

setup(
    name='sientia_tracker',
    version='1.0.6',
    license='Apache License 2.0',
    author=['Ítalo Azevedo', 'Pedro Bahia', 'Matheus Demoner'],
    author_email=['italo@aignosi.com.br', 'pedro.bahia@aignosi.com.br', 'matheus@aignosi.com.br'],
    description='Library for Aignosi Tracking API',
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords='sientia',
    packages=['sientia_tracker'],
    install_requires=['mlflow==2.10.1', 'typing'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)

