"""Import plugin code from `plugins/` files into the Markdown documentation file `plugins.md`."""

import ast
import difflib
import re
from pathlib import Path
from pprint import pprint


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

pattern = re.compile(r'```\w+\n# import from ([^\n]+)\n(.+?(?=\n```))', re.DOTALL)
class_name_pattern = re.compile(r'^class ([^\(]+)Plugin\(IPlugin\)', flags=re.MULTILINE)

readme = []

for m in reversed(list(pattern.finditer(plugins_doc))):
    filename = root_dir / m.group(1)
    plugin_contents = filename.read_text()
    i = m.start(2)
    j = m.end(2)
    plugins_doc = plugins_doc[:i] + plugin_contents + plugins_doc[j:]

# Build the README from every plugin file, including examples documented on
# their own how-to page rather than embedded in docs/plugins.md.
for filename in sorted((root_dir / 'plugins').glob('*.py')):
    plugin_contents = filename.read_text()
    match = class_name_pattern.search(plugin_contents)
    if not match:
        continue
    title = f'{match.group(1)}Plugin'
    desc = plugin_contents.splitlines()[0].replace('"', '')
    readme.append(f'* [{title}]({filename.name}): {desc}')

# Update the plugin README
readme_contents = '\n'.join(readme)
(root_dir / 'plugins/README.md').write_text(f'# phy plugin examples\n\n{readme_contents}\n')


# Make sure the copied and pasted code in the Markdown file is correct.
for m in pattern.finditer(plugins_doc):
    assert is_valid_python(m.group(2))
    filename = root_dir / m.group(1)
    plugin_contents = filename.read_text()
    assert plugin_contents.strip() == m.group(2).strip()


print('DIFF\n----\n')
a, b = plugins_doc0.splitlines(), plugins_doc.splitlines()
pprint('\n'.join([li for li in difflib.ndiff(a, b) if li[0] != ' ']))

plugins_file.write_text(plugins_doc)
print('Updated doc.')
