# -*- coding: utf-8 -*-

"""Minimal API documentation generation."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import re
import sys
import inspect


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


def _doc(obj):
    return inspect.getdoc(obj) or ''


def _import_module(module_name):
    """
    Imports a module. A single point of truth for importing modules to
    be documented by `pdoc`. In particular, it makes sure that the top
    module in `module_name` can be imported by using only the paths in
    `pdoc.import_path`.

    If a module has already been imported, then its corresponding entry
    in `sys.modules` is returned. This means that modules that have
    changed on disk cannot be re-imported in the same process and have
    its documentation updated.
    """
    import_path = sys.path[:]
    if import_path != sys.path:
        # Such a kludge. Only restrict imports if the `import_path` has
        # been changed. We don't want to always restrict imports, since
        # providing a path to `imp.find_module` stops it from searching
        # in special locations for built ins or frozen modules.
        #
        # The problem here is that this relies on the `sys.path` not being
        # independently changed since the initialization of this module.
        # If it is changed, then some packages may fail.
        #
        # Any other options available?

        # Raises an exception if the parent module cannot be imported.
        # This hopefully ensures that we only explicitly import modules
        # contained in `pdoc.import_path`.
        imp.find_module(module_name.split('.')[0], import_path)

    if module_name in sys.modules:
        return sys.modules[module_name]
    else:
        __import__(module_name)
        return sys.modules[module_name]


def _is_public(obj):
    name = _name(obj)
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
        return name.split('.')[0] == package
    return True


def _iter_doc_members(obj, package=None):
    for _, member in inspect.getmembers(obj):
        if _is_public(member):
            if package is None or _is_defined_in_package(member, package):
                yield member


#------------------------------------------------------------------------------
# Iteration functions
#------------------------------------------------------------------------------

def _iter_subpackages(package, subpackages):
    """Iterate through a list of subpackages."""
    for subpackage in subpackages:
        yield _import_module('{}.{}'.format(package, subpackage))


def _iter_vars(mod):
    """Iterate through a list of variables define in a module's
    public namespace."""
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


def _generate_paragraphs(package, subpackages):
    yield "# API documentation of {}".format(package)

    yield _doc(_import_module(package))

    yield "Here is the list of subpackages:"

    # Table of contents: list of modules.
    for subpackage in _iter_subpackages(package, subpackages):
        subpackage_name = subpackage.__name__

        yield "* [{name}](#{anchor})".format(name=subpackage_name,
                                             anchor=_anchor(subpackage_name),
                                             )

    yield ""

    # API doc of each module.
    for subpackage in _iter_subpackages(package, subpackages):
        subpackage_name = subpackage.__name__

        yield "## {}".format(subpackage_name)

        # Subpackage documentation.
        yield _doc(_import_module(subpackage_name))

        # List of top-level functions in the subpackage.
        for func in _iter_functions(subpackage):
            yield "`{}`".format(_full_name(subpackage, func))
            # TODO: func doc

        # All public classes.
        for klass in _iter_classes(subpackage):
            yield "### {}".format(_full_name(subpackage, klass))

            # Class documentation.
            yield _doc(klass)

            yield "#### Methods"

            for method in _iter_methods(klass, package):
                yield "`{}`".format(_full_name(subpackage, method))

                # Method documentation.
                yield _doc(method)

            yield "#### Properties"

            for property in _iter_properties(klass, package):
                yield "`{}`".format(_full_name(subpackage, property))

                # Property documentation.
                yield _doc(method)


def generate_api_doc(package, subpackages):
    for paragraph in _generate_paragraphs(package, subpackages):
        print(paragraph)
        if not paragraph.startswith('* '):
            print()


if __name__ == '__main__':

    package = 'phy'
    subpackages = ['cluster.manual',
                   'electrode',
                   'io',
                   'plot',
                   'stats',
                   'utils',
                   ]

    generate_api_doc(package, subpackages)
