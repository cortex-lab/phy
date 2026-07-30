"""Tests for correlogram controller interactions."""

from phy.cluster.views import CorrelogramView
from phy.plot.tests import mouse_click

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
        assert supervisor.selected_similar == [similar_cluster_id]

        view = gui.list_views(CorrelogramView)[0]
        qtbot.waitUntil(lambda: set(view.cluster_ids) == {0, similar_cluster_id})
        width, height = view.canvas.get_size()
        mouse_click(qtbot, view.canvas, (3 * width / 4, height / 4), button='Right')
        qtbot.waitUntil(lambda: similar_cluster_id in supervisor.selected_clusters)

        assert similar_cluster_id in supervisor.selected_clusters
        assert similar_cluster_id not in supervisor.selected_similar
    finally:
        gui.close()
        controller.close()
