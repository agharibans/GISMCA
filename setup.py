from setuptools import setup

setup(
	name = 'GISMCA',
	version = '0.2.0',
	description = 'Automated analysis of in vitro gastrointestinal smooth muscle contractions.',
	url = 'https://github.com/agharibans/GISMCA.git',
	author = 'Armen Gharibans',
	author_email = 'armen@alimetry.com',
    python_requires = "~=3.6.9",
    install_requires = ["numpy~=1.17.5","scipy~=1.4.1","matplotlib~=3.1.2","pandas~=0.25.3"],
	packages = ['GISMCA'],
	license = 'MIT',
	zip_safe = False
	)