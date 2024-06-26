# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

import sphinx.application

root_path = Path('..').resolve()

sys.path.insert(0, str(root_path/'src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Qualia-Plugin-SOM'
copyright = '2023, Pierre-Emmanuel Novac'
author = 'Pierre-Emmanuel Novac'

qualia_doc_base_url = 'https://leat-edge.github.io' if os.getenv('GITHUB_ACTIONS') else 'http://naixtech.unice.fr/~gitlab/docs'

# The full version, including alpha/beta/rc tags

def pdm_get_version(root_path: Path) -> str:
    import pdm.core
    import pdm.models.project_info

    core = pdm.core.Core()
    project = core.create_project(root_path=root_path)
    metadata = project.make_self_candidate(False).prepare(project.environment).prepare_metadata(True)
    project_info = pdm.models.project_info.ProjectInfo.from_distribution(metadata)

    return project_info.version

release = pdm_get_version(root_path=root_path)
version = release
#release = "alpha"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 5,
}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'PyTorch': ('https://pytorch.org/docs/stable/', None),
    'qualia_core': (f'{qualia_doc_base_url}/qualia-core', None),
    'qualia_codegen_core': (f'{qualia_doc_base_url}/qualia-codegen-core', None),
    'qualia_codegen_plugin_som': (f'{qualia_doc_base_url}/qualia-codegen-plugin-som', None),

}

show_authors = True

napoleon_use_ivar = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autodoc_mock_imports = []
autoclass_content = 'class'
autodoc_class_signature = 'separated'
autodoc_member_order = 'bysource'
autodoc_inherit_docstrings = False
autodoc_typehints = 'both'
autodoc_default_options = {'special-members': '__init__, __call__'}


master_doc = 'index'

# This is used by qualia_core.typing.TYPE_CHECKING to enable TYPE_CHECKING blocks for autodoc in order to get the most type
# references possible
os.environ['SPHINX_AUTODOC'] = '1'

# Call sphinx-apidoc
def run_apidoc(_: sphinx.application.Sphinx) -> int:
    from sphinx.ext.apidoc import main
    return main(['-e',
          '-o',
          str(Path().resolve()/'source'),
          str(root_path/'src'/'qualia_plugin_som'),
          '--force'])

def setup(app: sphinx.application.Sphinx) -> None:
    _ = app.connect('builder-inited', run_apidoc)
