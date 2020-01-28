from pathlib import Path
import re

from phy.apps.base import BaseController
from phy.cluster import views
from phy.cluster.supervisor import ActionCreator
from phy.gui.actions import _show_shortcuts, _show_snippets
from phy.gui import GUI
from phylib.utils.testing import captured_output


# Get a mapping view class : list of keyboard shortcuts

view_names = [v for v in dir(views) if v.endswith('View') and v != 'ManualClusteringView']
view_classes = [getattr(views, name) for name in view_names]


def _get_shortcuts(cls):
    with captured_output() as (stdout, stderr):
        print(cls.__name__)
        print('-' * len(cls.__name__))
        print()
        _show_shortcuts(cls.default_shortcuts)
        _show_snippets(cls.default_snippets)
    return stdout.getvalue()


view_shortcuts = {}
for cls in view_classes:
    s = _get_shortcuts(cls)
    if '-' in s:
        s = s[s.index('-') - 1:]
    view_shortcuts[cls.__name__] = s


# Insert the shortcuts in the Markdown files.
pattern = re.compile(r'```text\nKeyboard shortcuts for (\w+)\n([^`]+)\n```')
docs_dir = Path(__file__).parent.parent / 'docs/'
files = docs_dir.glob('*.md')
for file in files:
    contents = file.read_text()
    for m in reversed(list(pattern.finditer(contents))):
        view_name = m.group(1)
        shortcuts = view_shortcuts[view_name]
        i = m.start(2)
        j = m.end(2)
        contents = contents[:i] + shortcuts + contents[j:]
        file.write_text(contents)
        print("Inserted shortcuts for %s in %s." % (view_name, file))


# All shortcuts
supervisor_shortcuts = _get_shortcuts(ActionCreator).replace('ActionCreator', 'Clustering')
base_shortcuts = _get_shortcuts(BaseController)
gui_shortcuts = _get_shortcuts(GUI)

all_shortcuts = (
    supervisor_shortcuts + base_shortcuts + gui_shortcuts +
    ''.join(_get_shortcuts(cls) for cls in view_classes))

pattern = re.compile(r'```text\nAll keyboard shortcuts\n\n([^`]+)\n```')
shortcuts_file = docs_dir / 'shortcuts.md'
contents = shortcuts_file.read_text()
m = pattern.search(contents)
i = m.start(1)
j = m.end(1)
contents = contents[:i] + all_shortcuts + contents[j:]
shortcuts_file.write_text(contents)
print("Inserted all shortcuts in %s." % shortcuts_file)
