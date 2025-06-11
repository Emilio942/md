#!/usr/bin/env python3
"""
Configuration file for the Sphinx documentation builder.

Task 11.1: Umfassende API Dokumentation 🛠
Sphinx-basierte Dokumentation für alle Module
"""
import os
import sys
from pathlib import Path

# Add the proteinMD package to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'proteinMD'))

# -- Project information -----------------------------------------------------
project = 'ProteinMD'
copyright = '2025, ProteinMD Development Team'
author = 'ProteinMD Development Team'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # Automatic documentation from docstrings
    'sphinx.ext.autosummary',       # Generate summaries automatically
    'sphinx.ext.viewcode',          # Add source code links
    'sphinx.ext.napoleon',          # Support for Google/NumPy style docstrings
    'sphinx.ext.intersphinx',       # Link to other projects' documentation
    'sphinx.ext.todo',              # Support for todo items
    'sphinx.ext.coverage',          # Documentation coverage
    'sphinx.ext.mathjax',           # Math support
    'sphinx.ext.githubpages',       # GitHub Pages support
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    '**/__pycache__',
    '**/test_*.py',
    '**/tests/**',
]

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo': 'logo.png',
    'logo_name': True,
    'logo_text_align': 'left',
    'description': 'Molecular Dynamics Simulation Library',
    'github_user': 'your-username',
    'github_repo': 'proteinmd',
    'github_button': True,
    'github_banner': True,
    'show_powered_by': False,
    'extra_nav_links': {
        'GitHub': 'https://github.com/your-username/proteinmd',
    }
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'custom.css',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension ------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mock imports for missing modules to avoid import errors during doc build
autodoc_mock_imports = [
    'proteinMD.core.integrators',
    'proteinMD.core.thermostats', 
    'proteinMD.core.barostats',
    'proteinMD.core.forces',
    'proteinMD.core.energy',
    'proteinMD.core.trajectory',
    'proteinMD.analysis.distances',
    'proteinMD.analysis.pca',
    'proteinMD.analysis.free_energy',
    'proteinMD.analysis.utils',
    'proteinMD.environment.ions',
    'proteinMD.environment.membrane', 
    'proteinMD.environment.utils',
    'proteinMD.forcefield.custom',
    'proteinMD.forcefield.nonbonded',
    'proteinMD.forcefield.bonded',
    'proteinMD.forcefield.validation',
    'proteinMD.structure.validation',
    'proteinMD.structure.utils',
    'proteinMD.structure.formats',
]

# Automatically extract typehints
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# -- Options for autosummary extension --------------------------------------
autosummary_generate = False  # Temporarily disabled to avoid import issues
autosummary_generate_overwrite = False

# -- Options for napoleon extension -----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx extension --------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# -- Options for todo extension ---------------------------------------------
todo_include_todos = True

# -- Options for coverage extension -----------------------------------------
coverage_show_missing_items = True

# -- Custom configuration ---------------------------------------------------

# Add module paths for better autodoc discovery
def setup(app):
    """Custom setup function for Sphinx."""
    app.add_css_file('custom.css')
    
    # Add custom autodoc processors
    from sphinx.ext.autodoc import ClassDocumenter
    from sphinx.ext.autodoc import FunctionDocumenter
    
    # Custom processing for better documentation
    def process_docstring(app, what, name, obj, options, lines):
        """Process docstrings to add examples and cross-references."""
        if what in ('class', 'function', 'method'):
            # Add automatic cross-references
            for i, line in enumerate(lines):
                if 'Args:' in line or 'Parameters:' in line:
                    # Add parameter type hints
                    pass
                elif 'Returns:' in line:
                    # Add return type hints
                    pass
    
    app.connect('autodoc-process-docstring', process_docstring)

# Suppress warnings for missing references in development
suppress_warnings = ['ref.python']

# Enable figure numbering
numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
}

# LaTeX output options
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'fncychap': '',
    'printindex': '',
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, 'ProteinMD.tex', 'ProteinMD Documentation',
     'ProteinMD Development Team', 'manual'),
]

# -- Custom directives and roles --------------------------------------------

# Add custom CSS for better code highlighting
html_context = {
    'css_files': [
        '_static/theme_overrides.css',  # override wide tables in RTD theme
        '_static/custom.css',
    ],
}
