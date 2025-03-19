from setuptools import setup, find_packages

with open("README.md", "r") as arq:
    readme = arq.read()

setup(
    name='sientia_tracker',
    version='1.0.7',
    license='Apache License 2.0',
    author=['Ítalo Azevedo', 'Pedro Bahia', 'Matheus Demoner'],
    author_email=['italo@aignosi.com.br', 'pedro.bahia@aignosi.com.br', 'matheus@aignosi.com.br'],
    description='A Python library for tracking experiments and models on the SIENTIA edge AIOps platform.',
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords='sientia, aiops, tracking, machine learning, data science',
    packages=find_packages(),
    install_requires=[
        'mlflow==2.10.1',
        'typing',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    project_urls={
        'Documentation': 'https://aignosi.github.io/sientia-log-library/',
        'Source': 'https://github.com/aignosi/sientia-log-library',
        'Tracker': 'https://github.com/aignosi/sientia-log-library/issues',
    },
    python_requires='>=3.7',
)