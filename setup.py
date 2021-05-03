import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='simcse',
    version='0.0.1',
    url='https://github.com/princeton-nlp/SimCSE',
    packages=find_packages(where="src"),
    license='MIT',
    long_description=readme,
)