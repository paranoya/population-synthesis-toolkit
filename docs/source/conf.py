# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../src'))
    )


# Avoid build failures on read the docs
try:
    import numpy
except ImportError:
    from mock import MagicMock

    MOCK_MODULES = [
        'numpy',
        'astropy',
        'astropy.convolution',
        'astropy.io',
        'astropy.wcs',
        'matplotlib',
        'matplotlib.colors',
        'matplotlib.pyplot',
        'synphot',
        'scipy',
        'scipy.ndimage',
        'scipy.ndimage.interpolation',
        'scipy.optimize',
        'scipy.signal',
        "past",
        "past.utils"
    ]
    sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)



# -- Project information -----------------------------------------------------

project = 'Population Synthesis Toolkit'
copyright = '2024, Ascasibar Y., Corcho-Caballero P., Jiménez-López D.'
author = 'Ascasibar Y., Corcho-Caballero P., Jiménez-López D.'


import pst
version = '.'.join(pst.__version__.split(".")[0:2])
# The full version, including alpha/beta/rc tags.
release = pst.__version__

source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',

]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('https://docs.python.org/3/', None),
                       'astropy': ('https://docs.astropy.org/en/stable/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None)}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
autodoc_member_order = 'bysource'

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True