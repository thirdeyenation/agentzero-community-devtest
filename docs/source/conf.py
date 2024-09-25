# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Agent Zero Dev.Test Branch'
copyright = '2024, Dylan Radske (Agent Zero Community)'
author = 'Dylan Radske (Agent Zero Community)'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Assuming your Agent Zero code is in the parent directory

extensions = [
    # ...
    'sphinx_versions',
]

# HTML theme options
html_theme_options = {
    # ...
    'versions_dropdown': True,
}

extensions = [
    'sphinx.ext.autodoc',  # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Links to source code
]

html_theme = 'sphinx_rtd_theme'  # Use the Read the Docs theme