from setuptools import setup, find_packages


setup(
    name="pst",
    version='0.1',
    description='Population Synthesis Toolkit',
    long_description='This is a first test of the Python population synthesis package',
    long_description_content_type="text/markdown",
    include_package_data=True,
    author='Yago Ascasibar',
    author_email='yago.ascasibar@uam.es',
    url = 'https://github.com/paranoya/population-synthesis-toolkit',
    # download_url='https://github.com/PabloMSanAla/fabada/archive/refs/tags/v_01.tar.gz',
    packages=['pst'],
    python_requires='>=3.5',
    keywords=['Astronomy'],
    install_requires=['numpy', 'matplotlib'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: Free for non-commercial use',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules']
)
