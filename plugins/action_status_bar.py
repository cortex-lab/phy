"""Show how to create new actions in the GUI.

The first action just displays a message in the status bar.

The second action selects the first N clusters, where N is a parameter that is entered by
the user in a prompt dialog.

"""

from phy import IPlugin, connect


class ExampleActionPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):

            # Add a separator at the end of the File menu.
            # Note: currently, there is no way to add actions at another position in the menu.
            gui.file_actions.separator()

            # Add a new action to the File menu.
            @gui.file_actions.add(shortcut='a')  # the keyboard shortcut is A
            def display_message():
                """Display Hello world in the status bar."""
                # This docstring will be displayed in the status bar when hovering the mouse over
                # the menu item.

                # We update the text in the status bar.
                gui.status_message = "Hello world"

            # We add a separator at the end of the Select menu.
            gui.select_actions.separator()

            # Add an action to a new submenu called "My submenu". This action displays a prompt
            # dialog with the default value 10.
            @gui.select_actions.add(
                submenu='My submenu', shortcut='ctrl+c', prompt=True, prompt_default=lambda: 10)
            def select_n_first_clusters(n_clusters):

                # All cluster view methods are called with a callback function because of the
                # asynchronous nature of Python-Javascript interactions in Qt5.
                @controller.supervisor.cluster_view.get_ids
                def get_cluster_ids(cluster_ids):
                    """This function is called when the ordered list of cluster ids is returned
                    by the Javascript view."""

                    # We select the first n_clusters clusters.
                    controller.supervisor.select(cluster_ids[:n_clusters])
