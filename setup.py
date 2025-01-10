from setuptools import setup, find_packages
import sys
import warnings

requirements = [
    'torch',
    'transformers',
    'sentencepiece',
    'protobuf',
    # for hf backend
    'accelerate', 
    'bitsandbytes',
    # trie
    'numba'
]

if sys.platform == 'darwin':
    warnings.warn('vllm is not supported on macOS/Darwin platforms. Some features may not be available.')
else:
    requirements.append('vllm')

test_requirements = [
    'pytest',
    'pytest-benchmark',
    'arsenal @ git+https://github.com/timvieira/arsenal',
    'datasets', # for wikitext corpus
    'viztracer', # for profiling
    'IPython' # missing dep in arsenal
]

docs_requirements = [
    'mkdocs',
    'mkdocstrings[python]',
    'mkdocs-material',
    'mkdocs-gen-files',
    'mkdocs-literate-nav',
    'mkdocs-section-index',
]

setup(
    name='genlm-backend',
    version='0.0.1',
    description='',
    install_requires=requirements,
    extras_require={'test' : test_requirements, 'docs' : docs_requirements},
    python_requires='>=3.10',
    authors=['Ben LeBrun'],
    readme='README.md',
    scripts=[],
    packages=find_packages(),
)