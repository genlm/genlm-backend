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

setup(
    name='genlm-backend',
    version='0.0.1',
    description='',
    install_requires=requirements,
    extras_require={'test' : test_requirements},
    python_requires='>=3.10',
    authors=['Ben LeBrun'],
    readme='',
    scripts=[],
    packages=find_packages(),
)