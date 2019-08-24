# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------

import re
import logging
from pathlib import Path


log = logging.getLogger(__name__)


def _find(filename):
    return Path(__file__).parent.parent / 'glsl' / filename


def remove_comments(code):
    """ Remove C-style comment from GLSL code string """

    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*\n)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def do_replace(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return ""  # so we will return empty to remove the comment
        else:  # otherwise, we will return the 1st group
            return match.group(1)  # captured quoted-string

    return regex.sub(do_replace, code)


def remove_version(code):
    """ Remove any version directive """

    pattern = r'\#\s*version[^\r\n]*\n'
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
    return regex.sub('\n', code)


def merge_includes(code):
    """ Merge all includes recursively """

    # pattern = '\#\s*include\s*"(?P<filename>[a-zA-Z0-9\-\.\/]+)"[^\r\n]*\n'
    pattern = r'\#\s*include\s*"(?P<filename>[a-zA-Z0-9\-\.\/]+)"'
    regex = re.compile(pattern)
    includes = []

    def replace(match):
        filename = match.group("filename")

        if filename not in includes:
            includes.append(filename)
            path = _find(filename)
            if not path:
                log.critical('"%s" not found' % filename)
                raise RuntimeError("File not found")
            text = '\n// --- start of "%s" ---\n' % filename
            with open(str(path)) as f:
                text += remove_comments(f.read())
            text += '// --- end of "%s" ---\n' % filename
            return text
        return ''

    # Limit recursion to depth 10
    for i in range(10):
        if re.search(regex, code):
            code = re.sub(regex, replace, code)
        else:
            break

    return code


def preprocess(code):
    """ Preprocess a code by removing comments, version and merging includes"""

    if code:
        # code = remove_comments(code)
        code = remove_version(code)
        code = merge_includes(code)
    return code


def get_declarations(code, qualifier=""):
    """ Extract declarations of type:

        qualifier type name[,name,...];
    """

    if not len(code):
        return []

    variables = []

    if isinstance(qualifier, list):
        qualifier = "(" + "|".join([str(q) for q in qualifier]) + ")"

    if qualifier != "":
        re_type = re.compile(r"""
                             %s                               # Variable qualifier
                             \s+(?P<type>\w+)                 # Variable type
                             \s+(?P<names>[\w,\[\]\n =\.$]+); # Variable name(s)
                             """ % qualifier, re.VERBOSE)
    else:
        re_type = re.compile(r"""
                             \s*(?P<type>\w+)         # Variable type
                             \s+(?P<names>[\w\[\] ]+) # Variable name(s)
                             """, re.VERBOSE)

    re_names = re.compile(r"""
                          (?P<name>\w+)           # Variable name
                          \s*(\[(?P<size>\d+)\])? # Variable size
                          (\s*[^,]+)?
                          """, re.VERBOSE)

    for match in re.finditer(re_type, code):
        vtype = match.group('type')
        names = match.group('names')
        for match in re.finditer(re_names, names):
            name = match.group('name')
            size = match.group('size')
            if size is None:
                variables.append((name, vtype))
            else:
                size = int(size)
                if size == 0:
                    raise RuntimeError(
                        "Size of a variable array cannot be zero")
                for i in range(size):
                    iname = '%s[%d]' % (name, i)
                    variables.append((iname, vtype))
    return variables


def get_hooks(code):
    if not len(code):
        return []

    hooks = []
    re_hooks = re.compile(r"""\<(?P<hook>\w+)
                              (\.(?P<subhook>.+))?
                              (\([^<>]+\))?\>""", re.VERBOSE)
    for match in re.finditer(re_hooks, code):
        hooks.append((match.group('hook'), None))
    return list(set(hooks))


def get_args(code):
    return get_declarations(code, qualifier="")


def get_externs(code):
    return get_declarations(code, qualifier="extern")


def get_consts(code):
    return get_declarations(code, qualifier="const")


def get_uniforms(code):
    return get_declarations(code, qualifier="uniform")


def get_attributes(code):
    return get_declarations(code, qualifier=["attribute", "in"])


def get_varyings(code):
    return get_declarations(code, qualifier="varying")


def get_functions(code):
    def brace_matcher(n):
        # From stack overflow: python-how-to-match-nested-parentheses-with-regex
        # poor man's matched brace scanning, gives up
        # after n+1 levels.  Matches any string with balanced
        # braces inside; add the outer braces yourself if needed.
        # Nongreedy.
        return r"[^{}]*?(?:{" * n + r"[^{}]*?" + r"}[^{}]*?)*?" * n

    functions = []
    regex = re.compile(r"""
                       \s*(?P<type>\w+)    # Function return type
                       \s+(?P<name>[\w]+)   # Function name
                       \s*\((?P<args>.*?)\) # Function arguments
                       \s*\{(?P<code>%s)\} # Function content
                       """ % brace_matcher(5), re.VERBOSE | re.DOTALL)

    for match in re.finditer(regex, code):
        rtype = match.group('type')
        name = match.group('name')
        args = match.group('args')
        fcode = match.group('code')
        if name not in ("if", "while"):
            functions.append((rtype, name, args, fcode))
    return functions


def parse(code):
    """ Parse a shader """

    code = preprocess(code)
    externs = get_externs(code) if code else []
    consts = get_consts(code) if code else []
    uniforms = get_uniforms(code) if code else []
    attributes = get_attributes(code) if code else []
    varyings = get_varyings(code) if code else []
    hooks = get_hooks(code) if code else []
    functions = get_functions(code) if code else []

    return {'externs': externs,
            'consts': consts,
            'uniforms': uniforms,
            'attributes': attributes,
            'varyings': varyings,
            'hooks': hooks,
            'functions': functions}
