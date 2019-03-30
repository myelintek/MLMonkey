import os.path
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# Get current __version__
version_locals = {}
execfile(os.path.join(here, 'mlmonkey', 'version.py'), {}, version_locals)

# Get requirements
requirements = []
with open(os.path.join(here, 'requirements.txt'), 'r') as infile:
    for line in infile:
        line = line.strip()
        if line and not line[0] == '#':  # ignore comments
            requirements.append(line)

setuptools.setup(
    name='mlmonkey',
    version=version_locals['__version__'],
    description="Deep Learning GPU benchmark",
    license='MIT',
    classifiers=[
        'Framework :: Flask',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 2 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='benchmark',
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    entry_points={  # Optional
        'console_scripts': [
            'runserver = mlmonkey.app:main',
        ],
    },
)
