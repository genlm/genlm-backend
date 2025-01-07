from setuptools import setup, find_packages

requirements = [
    'vllm',
    'torch',
    'transformers',
    # for hf backend
    'accelerate', 
    'bitsandbytes',
    # trie
    'numba'
]

test_requirements = [
    'pytest',
    'arsenal @ git+https://github.com/timvieira/arsenal',
    'datasets', # for wikitext corpus
    'viztracer', # for profiling
    'pytest-benchmark',
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
    readme='',
    scripts=[],
    packages=find_packages(),
)