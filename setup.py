from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')



setup(
	name='phenobert',
	version='1.0.0',
	description='A novel tool for human clinical disease phenotype recognizing with deep learning.', 
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/EclipseCN/PhenoBERT',
    author='NeoFengyh',
	author_email='18210700100@fudan.edu.cn',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Bioinformatics Engineers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
		keywords='deep-learning, nlp, bert', 
    ],
    url='https://github.com/YaokaiYang-assaultmaster/py3PortScanner',
    packages=find_packages(),
    package_data={'pyportscanner': ['etc/*.dat']},
    include_package_data=True,
    zip_safe=False,
    extras_require={
      'dev': dev_requires,
    },
)