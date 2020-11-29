import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

__VERSION__ = "0.1.0"

setuptools.setup(
	name="povel_essens",
	version=__VERSION__,
	py_modules=["povel_essens.cscore"],
	author="Luke Poeppel",
	author_email="luke.poeppel@gmail.com",
	description="Implementation of Povel & Essens' algorithm for generating the C-score of a rhythm.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/Luke-Poeppel/povel_essens",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.7',
)