"""Hello world plugin."""

from phy import IPlugin, connect


class ExampleHelloPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect(sender=controller)
        def on_gui_ready(sender, gui):
            """This is called when the GUI and all objects are fully loaded.
            This is to make sure that controller.supervisor is properly defined.
            """

            @connect(sender=controller.supervisor)
            def on_cluster(sender, up):
                """This is called every time a cluster assignment or cluster group/label
                changes."""
                print("Clusters update: %s" % up)
