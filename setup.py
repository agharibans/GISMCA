from setuptools import setup

setup(
	name = 'GISMCA',
	version = '0.4.1',
	description = 'Automated analysis of in vitro gastrointestinal smooth muscle contractions.',
	url = 'https://github.com/agharibans/GISMCA.git',
	author = 'Armen Gharibans',
	author_email = 'armen@alimetry.com',
    python_requires = "~=3.6.9",
    install_requires = ["numpy~=1.17","scipy~=1.4","matplotlib~=3.1","pandas~=1.0","scikit-image~=0.16"],
	packages = ['GISMCA'],
	license = 'MIT',
	zip_safe = False
	)