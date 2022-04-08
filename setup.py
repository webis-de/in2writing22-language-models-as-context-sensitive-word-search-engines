from setuptools import setup, find_packages

setup(
    name='lcwp',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'wheel', 'Click', 'tqdm', 'plotly', 'spacy>=3.0.0,<4.0.0', "sonora", "grpcio>=1.43.0", "protobuf", "torch", "transformers", "datasets"
    ],
    entry_points='''
        [console_scripts]
        main=lcwp.main:cli
    ''',
)