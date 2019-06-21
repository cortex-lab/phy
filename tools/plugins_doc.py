# -*- coding: utf-8 -*-
"""Import plugin code from `plugins/` files into the Markdown documentation file `plugins.md`."""

import ast
import difflib
from pathlib import Path
from pprint import pprint
import re


def is_valid_python(code):
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


plugins_file = Path(__file__).parent / '../docs/plugins.md'
root_dir = Path(__file__).parent.parent

plugins_doc = plugins_file.read_text()
plugins_doc0 = plugins_doc

pattern = re.compile(r'```\w+\n# import from ([^\n]+)\n([^`]+)\n```')

for m in reversed(list(pattern.finditer(plugins_doc))):
    filename = root_dir / m.group(1)
    plugin_contents = filename.read_text()
    i = m.start(2)
    j = m.end(2)
    plugins_doc = plugins_doc[:i] + plugin_contents + plugins_doc[j:]

# Make sure the copied and pasted code in the Markdown file is correct.
for m in pattern.finditer(plugins_doc):
    assert is_valid_python(m.group(2))
    filename = root_dir / m.group(1)
    plugin_contents = filename.read_text()
    assert plugin_contents.strip() == m.group(2).strip()


print("DIFF\n----\n")
a, b = plugins_doc0.splitlines(), plugins_doc.splitlines()
pprint('\n'.join([li for li in difflib.ndiff(a, b) if li[0] != ' ']))

plugins_file.write_text(plugins_doc)
print("Updated doc.")
