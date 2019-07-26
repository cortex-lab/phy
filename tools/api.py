# -*- coding: utf-8 -*-
"""Minimal API documentation generation."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from importlib import import_module
import inspect
import os.path as op
import re


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _name(obj):
    if hasattr(obj, '__name__'):
        return obj.__name__
    elif inspect.isdatadescriptor(obj):
        return obj.fget.__name__


def _full_name(subpackage, obj):
    return '{}.{}'.format(subpackage.__name__, _name(obj))


def _anchor(name):
    anchor = name.lower().replace(' ', '-')
    anchor = re.sub(r'[^\w\- ]', '', anchor)
    return anchor


_docstring_header_pattern = re.compile(r'^([^\n]+)\n[\-\=]{3,}$', flags=re.MULTILINE)
_docstring_parameters_pattern = re.compile(r'^([^ \n]+) \: ([^\n]+)$', flags=re.MULTILINE)


def _replace_docstring_header(paragraph):
    """Process NumPy-like function docstrings."""
    # Replace Markdown headers in docstrings with light headers in bold.
    paragraph = re.sub(_docstring_header_pattern, r'**\1**', paragraph)
    paragraph = re.sub(_docstring_parameters_pattern, r'\n* `\1 : \2` 　 ', paragraph)
    return paragraph


def _doc(obj):
    doc = inspect.getdoc(obj) or ''
    doc = doc.strip()
    if r'\n\n' in doc:
        i = doc.index(r'\n\n')
        doc[:i] = re.sub(r'\n(?!=\n)', '', doc[:i])  # remove standalone newlines
    if doc and '---' in doc:
        return _replace_docstring_header(doc)
    else:
        return doc


#------------------------------------------------------------------------------
# Introspection methods
#------------------------------------------------------------------------------

def _is_public(obj):
    name = _name(obj) if not isinstance(obj, str) else obj
    if name:
        return not name.startswith('_')
    else:
        return True


def _is_defined_in_package(obj, package):
    if isinstance(obj, property):
        obj = obj.fget
    mod = inspect.getmodule(obj)
    if mod and hasattr(mod, '__name__'):
        name = mod.__name__
        return name.split('.')[0].startswith(package)
    return True


def _iter_doc_members(obj, package=None):
    for name, member in inspect.getmembers(obj):
        if _is_public(name):
            if package is None or _is_defined_in_package(member, package):
                yield member


def _iter_subpackages(package, subpackages):
    """Iterate through a list of subpackages."""
    for subpackage in subpackages:
        yield import_module('{}.{}'.format(package, subpackage))


def _iter_vars(mod):
    """Iterate through a list of variables define in a module's public namespace."""
    vars = sorted(var for var in dir(mod) if _is_public(var))
    for var in vars:
        yield getattr(mod, var)


def _iter_functions(subpackage):
    return filter(inspect.isfunction, _iter_vars(subpackage))


def _iter_classes(subpackage):
    return filter(inspect.isclass, _iter_vars(subpackage))


def _iter_methods(klass, package=None):
    for member in _iter_doc_members(klass, package):
        if inspect.isfunction(member) or inspect.ismethod(member):
            if inspect.isdatadescriptor(member):
                continue
            yield member


def _iter_properties(klass, package=None):
    for member in _iter_doc_members(klass, package):
        if isinstance(member, property):
            yield member.fget


#------------------------------------------------------------------------------
# API doc generation
#------------------------------------------------------------------------------

def _function_header(subpackage, func):
    """Generate the docstring of a function."""
    args = str(inspect.signature(func))
    return "{name}{args}".format(name=_full_name(subpackage, func), args=args)


_FUNCTION_PATTERN = '%s\n\n\n**`%s`**\n\n%s\n\n---'


def _doc_function(subpackage, func):
    title = _full_name(subpackage, func)
    return _FUNCTION_PATTERN % (title, _function_header(subpackage, func), _doc(func))


def _doc_method(klass, func):
    """Generate the docstring of a method."""
    args = str(inspect.signature(func))
    title = "{klass}.{name}".format(klass=klass.__name__, name=_name(func))
    header = "{klass}.{name}{args}".format(klass=klass.__name__, name=_name(func), args=args)
    docstring = _doc(func)
    return _FUNCTION_PATTERN % (title, header, docstring)


def _doc_property(klass, prop):
    """Generate the docstring of a property."""
    header = "{klass}.{name}".format(klass=klass.__name__, name=_name(prop))
    docstring = _doc(prop)
    return _FUNCTION_PATTERN % (header, header, docstring)


def _link(name, anchor=None):
    return "[{name}](#{anchor})".format(name=name, anchor=anchor or _anchor(name))


def _generate_preamble(package, subpackages):

    yield "# API documentation of {}".format(package)

    yield _doc(import_module(package))

    yield "## Table of contents"

    # Table of contents: list of modules.
    for subpackage in _iter_subpackages(package, subpackages):
        subpackage_name = subpackage.__name__

        yield "### " + _link(subpackage_name)

        # List of top-level functions in the subpackage.
        for func in _iter_functions(subpackage):
            yield '* ' + _link(
                _full_name(subpackage, func), _anchor(_full_name(subpackage, func)))

        # All public classes.
        for klass in _iter_classes(subpackage):

            # Class documentation.
            yield "* " + _link(_full_name(subpackage, klass))

        yield ""

    yield ""


def _generate_paragraphs(package, subpackages):
    """Generate the paragraphs of the API documentation."""

    # API doc of each module.
    for subpackage in _iter_subpackages(package, subpackages):
        subpackage_name = subpackage.__name__

        yield "## {}".format(subpackage_name)

        # Subpackage documentation.
        yield _doc(import_module(subpackage_name))

        yield "---"

        # List of top-level functions in the subpackage.
        for func in _iter_functions(subpackage):
            yield '#### ' + _doc_function(subpackage, func)

        # All public classes.
        for klass in _iter_classes(subpackage):

            # Class documentation.
            yield "### {}".format(_full_name(subpackage, klass))
            yield _doc(klass)

            yield "---"

            for method in _iter_methods(klass, package):
                yield '#### ' + _doc_method(klass, method)

            for prop in _iter_properties(klass, package):
                yield '#### ' + _doc_property(klass, prop)


def _print_paragraph(paragraph):
    out = ''
    out += paragraph + '\n'
    if not paragraph.startswith('* '):
        out += '\n'
    return out


def generate_api_doc(package, subpackages, path=None):
    out = ''
    for paragraph in _generate_preamble(package, subpackages):
        out += _print_paragraph(paragraph)
    for paragraph in _generate_paragraphs(package, subpackages):
        out += _print_paragraph(paragraph)
    if path is None:
        return out
    else:
        with open(path, 'w') as f:
            f.write('\n'.join([_.rstrip() for _ in out.splitlines()]))


if __name__ == '__main__':

    package = 'phy'
    subpackages = ['utils', 'gui', 'plot', 'cluster', 'apps', 'apps.template', 'apps.kwik']

    curdir = op.dirname(op.realpath(__file__))
    path = op.join(curdir, '../docs/api.md')
    generate_api_doc(package, subpackages, path=path)
