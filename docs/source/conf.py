import os
import sys
from datetime import datetime

# Add repo root to sys.path so autodoc can import ptyrax
sys.path.insert(0, os.path.abspath("../../"))

project = "Ptyrax"
author = "Sander Senhorst"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    # Avoid sphinx_autodoc_typehints: it evaluates complex jaxtyping
    # signatures and can raise during docs builds. The built-in
    # `sphinx.ext.autodoc`/typehints behaviour is sufficient here.
]

# Prevent duplicate attribute object declarations when class docstrings include
# an "Attributes" section and autodoc also renders annotated class fields.
napoleon_use_ivar = True
napoleon_attr_annotations = False

source_suffix = [".rst", ".ipynb", ".md"]

language = "en"

exclude_patterns = [
    # Sometimes sphinx reads its own outputs as inputs!
    "_build/html",
    "_build/jupyter_execute",
]

autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_typehints_format = "short"
# Default options for autodoc. Keep the output concise but include useful
# metadata (signature, typehints, show inheritance) and automatically list
# members. Tests/docs often import heavy optional deps (jax, equinox, vispy),
# so mock them when the environment does not provide them to avoid build
# failures in CI or minimal environments.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
    "special-members": "__call__, __plot__",
}
remove_from_toctrees = ["_autosummary/*"]
# autodoc_mock_imports = [
#     "jaxtyping",
# ]
autodoc_type_aliases = {
    "array": "Array",
    "int": "jaxtyping.Integer",
    "float": "jaxtyping.Float",
}

# myst-nb / myst-parser configuration
nb_execution_mode = "off"
nb_parse_markdown = True
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]

# Some tutorial notebooks intentionally contain long traceback outputs that
# include mixed formatting tokens; suppress lexer warnings for those blocks.
suppress_warnings = ["misc.highlighting_failure"]

# Do not execute notebooks by default in CI — set to 'auto' locally if desired
# jupyter_execute_notebooks = "auto"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_theme_options = {
    "show_toc_level": 2,
    # repository_url can be set to your project's repository
    "repository_url": "https://github.com/ssenhorst/ptyrax",
    "use_repository_button": True,
    "navigation_with_keys": False,
    "article_header_start": ["toggle-primary-sidebar.html", "breadcrumbs"],
}

# Path for static assets (logo, favicon, css)
html_static_path = ["_static"]

# Optional theme assets
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"
html_css_files = [
    "style.css",
]
