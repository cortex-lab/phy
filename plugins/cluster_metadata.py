"""Show how to save the best channel of every cluster in a cluster_channel.tsv file when saving.

Note: this information is also automatically stored in `cluster_info.tsv` natively in phy,
along with all values found in the GUI cluster view.

"""

import logging

from phy import IPlugin, connect
from phylib.io.model import save_metadata

logger = logging.getLogger('phy')


class ExampleClusterMetadataPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):

            @connect(sender=gui)
            def on_request_save(sender):
                """This function is called whenever the Save action is triggered."""

                # We get the filename.
                filename = controller.model.dir_path / 'cluster_channel.tsv'

                # We get the list of all clusters.
                cluster_ids = controller.supervisor.clustering.cluster_ids

                # Field name used in the header of the TSV file.
                field_name = 'channel'

                # NOTE: cluster_XXX.tsv files are automatically loaded in phy, displayed
                # in the cluster view, and interpreted as cluster labels, *except* if their
                # name conflicts with an existing built-in column in the cluster view.
                # This is the case here, because there is a default channel column in phy.
                # Therefore, the TSV file is properly saved, but it is not displayed in the
                # cluster view as the information is already shown in the built-in channel column.
                # If you want this file to be loaded in the cluster view, just use another
                # name that is not already used, like 'best_channel'.

                # Dictionary mapping cluster_ids to the best channel id.
                metadata = {
                    cluster_id: controller.get_best_channel(cluster_id)
                    for cluster_id in cluster_ids}

                # Save the metadata file.
                save_metadata(filename, field_name, metadata)
                logger.info("Saved %s.", filename)
