from setuptools import setup


def readme():
	with open('Readme.md') as f:
		return f.read()

setup(
	name='Bioto',
	version='0.1.2',
	description='Collection of tools for network analyses',
	long_description=readme(),
	url='https://github.com/kpj/Bioto',
	author='kpj',
	author_email='kpjkpjkpjkpjkpjkpj@gmail.com',
	license='MIT',
	packages=[],
	test_suite='nose.collector',
	tests_require=['nose'],
	scripts=[],
	requires=['numpy', 'networkx', 'sympy', 'matplotlib', 'pandas', 'scipy', 'pysoft', 'beautifulsoup4', 'progressbar2', 'prettytable']
)
