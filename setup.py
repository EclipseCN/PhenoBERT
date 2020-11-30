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
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
		],
	keywords='deep-learning, nlp, bert', 
	package_dir={'': 'PhenoBERT_src'}, 
	packages=find_packages(where='PhenoBERT_src'),
	include_package_data=True,
	python_requires='>=3.6, <4',
	install_requires=['lxml>=4.6.1', 'nltk>=3.5', 'prettytable>=1.0.1', 'fasttext>=0.9.2', 'torch>=1.3.1', 'scipy>=1.5.2', 'stanza>=1.1.1'],
)