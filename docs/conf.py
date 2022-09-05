# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from os import path
import packaging.version
import re
import sys


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, path.abspath('..'))
# Custom extensions are installed here by `tensorflow-manylinux`.
sys.path.insert(0, path.abspath('/opt/sphinx'))
sys.path.insert(0, path.abspath('/opt/sphinx/extensions'))


# -- Project information -----------------------------------------------------

ROOT = path.abspath(path.join(path.dirname(__file__), '..'))

ABOUT = {}
with open(path.join(ROOT, "tensorflow_nufft/__about__.py")) as f:
  exec(f.read(), ABOUT)
_version = packaging.version.Version(ABOUT['__version__'])

project = ABOUT['__title__']
copyright = ABOUT['__copyright__']
author = ABOUT['__author__']
release = _version.public
version = '.'.join(map(str, (_version.major, _version.minor)))


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_nb',
    # Custom extensions installed by `tensorflow-manylinux` in
    # `/opt/sphinx/extensions`.
    'myst_autodoc',
    'myst_autosummary',
    'myst_napoleon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Do not add full qualification to objects' signatures.
add_module_names = False

# Set language.
language = "en"


# -- Options for HTML output -------------------------------------------------

html_title = 'TensorFlow NUFFT Documentation'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# https://sphinx-book-theme.readthedocs.io/en/latest/tutorials/get-started.html
html_theme_options = {
    'repository_url': 'https://github.com/mrphys/tensorflow-nufft',
    'use_repository_button': True,
    'launch_buttons': {
        'colab_url': "https://colab.research.google.com/"
    },
    'path_to_docs': 'docs'
}

# -- Options for MyST ----------------------------------------------------------

# https://myst-nb.readthedocs.io/en/latest/authoring/jupyter-notebooks.html
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_image",
    "substitution"
]

# https://myst-nb.readthedocs.io/en/latest/authoring/basics.html
source_suffix = [
    '.rst',
    '.md',
    '.ipynb'
]

# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#substitutions-with-jinja2
myst_substitutions = {
    'release': release
}

# For verification in Google Search Console.
myst_html_meta = {
    "google-site-verification": "8PySedj6KJ0kc5qC1CbO6_9blFB9Nho3SgXvbRzyVOU"
}

# -- Options for autosummary -------------------------------------------------

autosummary_generate = True

import autosummary_filename_map as afm
autosummary_filename_map = afm.AutosummaryFilenameMap()


def process_docstring(app, what, name, obj, options, lines):
  """Process autodoc docstrings."""
  # Regular expressions.
  blankline_re = re.compile(r"^\s*$")
  prompt_re = re.compile(r"^\s*>>>")
  tf_symbol_re = re.compile(r"`(?P<symbol>tf\.[a-zA-Z0-9_.]+)`")
  tfft_symbol_re = re.compile(r"`(?P<symbol>tfft\.[a-zA-Z0-9_.]+)`")

  # Loop initialization. `insert_lines` keeps a list of lines to be inserted
  # as well as their positions.
  insert_lines = []
  in_prompt = False

  # Iterate line by line.
  for lineno, line in enumerate(lines):

    # Check if we're in a prompt block.
    if in_prompt:
      # Check if end of prompt block.
      if blankline_re.match(line):
        in_prompt = False
        insert_lines.append((lineno, "```"))
        continue

    # Check for >>> prompt, if found insert code block (unless already in
    # prompt).
    m = prompt_re.match(line)
    if m and not in_prompt:
      in_prompt = True
      # We need to insert a new line. It's not safe to modify the list we're
      # iterating over, so instead we store the line in `insert_lines` and we
      # insert it after the loop.
      insert_lines.append((lineno, "```python"))
      continue

    # Add links to TF symbols.
    m = tf_symbol_re.search(line)
    if m:
      symbol = m.group('symbol')
      link = f"https://www.tensorflow.org/api_docs/python/{symbol.replace('.', '/')}"
      lines[lineno] = line.replace(f"`{symbol}`", f"[`{symbol}`]({link})")

    # Add links to TFFT symbols.
    m = tfft_symbol_re.search(line)
    if m:
      symbol = m.group('symbol')
      link = f"https://mrphys.github.io/tensorflow-nufft/api_docs/{symbol.replace('.', '/')}"
      lines[lineno] = line.replace(f"`{symbol}`", f"[`{symbol}`]({link})")

  # Now insert the lines (in reversed order so that line numbers stay valid).
  for lineno, line in reversed(insert_lines):
    lines.insert(lineno, line)


def setup(app):
  app.connect('autodoc-process-docstring', process_docstring)
