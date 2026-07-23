"""Tests for correlogram controller interactions."""

from phylib.utils import emit

from phy.cluster.views import CorrelogramView

from .test_base import MyController, _mock_controller


def test_correlogram_right_click_promotes_similar_cluster(qtbot, tempdir):
    controller = _mock_controller(tempdir, MyController)
    gui = controller.create_gui(do_prompt_save=False)
    with qtbot.waitExposed(gui):
        gui.show()

    try:
        supervisor = controller.supervisor
        supervisor.select([0])
        supervisor.block()
        similar_cluster_id = supervisor.similarity_view.get_ids()[0]
        supervisor.similarity_view.select([similar_cluster_id])
        supervisor.block()

        view = gui.list_views(CorrelogramView)[0]
        emit('request_promote_similar', view, 0, similar_cluster_id)
        supervisor.block()

        assert similar_cluster_id in supervisor.selected_clusters
        assert similar_cluster_id not in supervisor.selected_similar
    finally:
        gui.close()
        controller.close()
