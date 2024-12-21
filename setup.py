from setuptools import setup

requirements = [
    'vllm',
    'torch',
    'transformers',
    # for hf backend
    'accelarate', 
    'bitsandbytes' 
]

test_requirements = [
    'pytest',
    'arsenal @ git+https://github.com/timvieira/arsenal',
    'datasets', # for wikitext corpus
    'viztracer', # for profiling
    'pandas'
]

setup(
    name='async-llm',
    version='0.0.1',
    description='',
    install_requires=requirements,
    extras_require={'test' : test_requirements},
    python_requires='>=3.10',
    authors=['Ben LeBrun'],
    readme='',
    scripts=[],
    packages=['async_llm'],
)