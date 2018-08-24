# from allensdk.eye_tracking.ui import qt, __main__
# from allensdk.eye_tracking._schemas import InputParameters
# from allensdk.eye_tracking.frame_stream import CvOutputStream
# import os
# import sys
# import numpy as np
# import json
# import pytest  # noqa: F401
# from mock import patch, MagicMock

# DEFAULT_CV_FRAMES = 20
# # H264 is not available by default on windows
# if sys.platform == "win32":
#     FOURCC = "FMP4"
# else:
#     FOURCC = "H264"


# def image(shape=(200, 200), value=0):
#     return np.ones(shape, dtype=np.uint8)*value*10


# @pytest.fixture(scope="module", params=[True, False])
# def movie(tmpdir_factory, request):
#     if not request.param:
#         return ""
#     frame = image()
#     filename = str(tmpdir_factory.mktemp("test").join('movie.avi'))
#     ostream = CvOutputStream(filename, frame.shape[::-1], is_color=False,
#                              fourcc=FOURCC)
#     ostream.open(filename)
#     for i in range(DEFAULT_CV_FRAMES):
#         ostream.write(image(value=i))
#     ostream.close()
#     return filename


# @pytest.fixture(scope="module", params=[True, False])
# def config_file(tmpdir_factory, request):
#     if request.param:
#         filename = str(tmpdir_factory.mktemp("test").join('config.json'))
#         config = {"input_json": {"input_source": {"read_only": True}}}
#         with open(filename, "w") as f:
#             json.dump(config, f)
#     else:
#         filename = ""
#     return filename


# @pytest.fixture(params=[True, False])
# def json_file(tmpdir_factory, request):
#     if request.param:
#         return str(tmpdir_factory.mktemp("test").join("file.json"))
#     else:
#         return ""


# @pytest.fixture(params=[True, False])
# def mock_file_event(movie, request):
#     mock_event = MagicMock()
#     mock_data = MagicMock()
#     mock_urls = MagicMock(return_value=[movie])
#     mock_urls.toLocalFile = MagicMock(return_value=movie)
#     mock_data.hasUrls = MagicMock(return_value=request.param)
#     mock_data.urls = MagicMock(return_value=mock_urls)
#     mock_event.mimeData = MagicMock(return_value=mock_data)
#     return mock_event


# @patch.object(qt.DropFileMixin, "file_dropped")
# def test_drop_file_mixin(mock_signal, qtbot, mock_file_event):
#     w = qt.DropFileMixin()
#     w.dragEnterEvent(mock_file_event)
#     if mock_file_event.mimeData().hasUrls():
#         mock_file_event.accept.assert_called_once()
#     else:
#         mock_file_event.ignore.assert_called_once()
#     mock_file_event.reset_mock()
#     w.dragMoveEvent(mock_file_event)
#     if mock_file_event.mimeData().hasUrls():
#         mock_file_event.accept.assert_called_once()
#     else:
#         mock_file_event.ignore.assert_called_once()
#     mock_file_event.reset_mock()
#     w.dropEvent(mock_file_event)
#     if mock_file_event.mimeData().hasUrls():
#         mock_file_event.accept.assert_called_once()
#     else:
#         mock_file_event.ignore.assert_called_once()
#     mock_file_event.reset_mock()


# def test_field_widget(qtbot):
#     schema = InputParameters()
#     w = qt.FieldWidget("test", schema.fields["output_dir"])
#     assert(w.key == "test")
#     assert(w.field == schema.fields["output_dir"])
#     js = w.get_json()
#     default = schema.fields["output_dir"].default
#     assert(os.path.normpath(js) == os.path.normpath(default))

#     w = qt.FieldWidget("test2", schema.fields["input_source"])
#     assert(w.key == "test2")
#     assert(str(w.text()) == "")
#     js = w.get_json()
#     assert(js is None)

#     w = qt.FieldWidget("test3", schema.fields["pupil_bounding_box"])
#     assert(w.key == "test3")
#     default = schema.fields["pupil_bounding_box"].default
#     assert(str(w.text()) == str(default))
#     js = w.get_json()
#     assert(js == default)


# def test_schema_widget(qtbot):
#     schema = InputParameters()
#     w = qt.SchemaWidget(None, schema, None)
#     assert(w.key is None)
#     assert(isinstance(w.fields["eye_params"], qt.SchemaWidget))
#     assert(isinstance(w.fields["input_source"], qt.FieldWidget))

#     js = w.get_json()
#     assert(js)
#     with patch.object(qt.FieldWidget, "get_json", return_value=None):
#         js = w.get_json()
#         assert(js is None)

#     w.update_value("eye_params.min_pupil_value", "1000")
#     val = str(w.fields["eye_params"].fields["min_pupil_value"].text())
#     assert(val == "1000")


# def test_input_json_widget(qtbot):
#     schema = InputParameters()
#     w = qt.InputJsonWidget(schema)
#     assert(isinstance(w.schema_widget, qt.SchemaWidget))
#     w.update_value("eye_params.min_pupil_value", "1000")
#     js = w.get_json()
#     assert(js["eye_params"]["min_pupil_value"] == 1000)


# def test_bbox_canvas(qtbot):
#     w = qt.BBoxCanvas(qt.Figure())
#     w.set_rgb(0, 0, 0)
#     assert(w.rgba == (0, 0, 0, 20))
#     assert(not w.drawing)
#     mock_wheel = MagicMock()
#     w.wheelEvent(mock_wheel)
#     mock_wheel.ignore.assert_called_once()
#     w.show()
#     w.move(0, 0)
#     qtbot.addWidget(w)
#     qtbot.mousePress(w, qt.QtCore.Qt.LeftButton,
#                      qt.QtCore.Qt.NoModifier,
#                      qt.QtCore.QPoint(0, 0))
#     assert(w.drawing)
#     qtbot.mouseMove(w, qt.QtCore.QPoint(30, 30), 100)
#     assert(w.drawing)
#     with qtbot.waitSignal(w.box_updated) as box_updated:
#         qtbot.mouseRelease(w, qt.QtCore.Qt.LeftButton,
#                            qt.QtCore.Qt.NoModifier,
#                            qt.QtCore.QPoint(50, 50))
#     assert(not w.drawing)
#     assert(box_updated.signal_triggered)


# @patch.object(qt.BBoxCanvas, "width", return_value=200)
# @patch.object(qt.BBoxCanvas, "height", return_value=100)
# def test_bbox_canvas_scaling(h, w, qtbot):
#     w = qt.BBoxCanvas(qt.Figure())
#     assert(w.im_shape == (100, 200))
#     w.im_shape = (100, 100)
#     assert(w.im_shape == (100, 100))
#     s, x, y = w._scale_and_offset()
#     assert(y == 0)
#     assert(np.allclose(s, 1.0))
#     w.im_shape = (100, 400)
#     assert(w.im_shape == (100, 400))
#     s, x, y = w._scale_and_offset()
#     assert(x == 0)
#     assert(np.allclose(s, 2.0))


# @pytest.mark.parametrize("profile,config", [
#     (True, {"input_json": {"cr_bounding_box": {"visible": False}}}),
#     (False, None)
# ])
# @patch.object(qt.QtWidgets.QMessageBox, "exec_")
# def test_viewer_window(mock_exec, qtbot, movie, json_file, profile, config):
#     schema_type = InputParameters
#     w = qt.ViewerWindow(schema_type, profile_runs=profile, config=config)
#     w.widget.json_view.update_value("cr_bounding_box", "[")
#     qtbot.mouseClick(w.widget.rerun_button, qt.QtCore.Qt.LeftButton)
#     mock_exec.assert_called_once()
#     mock_exec.reset_mock()
#     w.widget.save_json()
#     mock_exec.assert_called_once()
#     w.widget.json_view.update_value("cr_bounding_box", "[]")
#     w.widget._setup_bbox()
#     w.widget.update_bbox(100, 200, 100, 200)
#     qtbot.mouseClick(w.widget.pupil_radio, qt.QtCore.Qt.LeftButton)
#     w.widget.update_bbox(10, 50, 10, 50)
#     assert(w.widget.get_json_data()["pupil_bounding_box"] == [10, 50, 10, 50])
#     qtbot.mouseClick(w.widget.cr_radio, qt.QtCore.Qt.LeftButton)
#     w.widget.update_bbox(10, 50, 10, 50)
#     assert(w.widget.get_json_data()["cr_bounding_box"] == [10, 50, 10, 50])
#     with patch.object(qt.QtWidgets.QFileDialog, "getSaveFileName",
#                       return_value=(json_file, None)):
#         mock_exec.reset_mock()
#         w.widget.save_json()
#         mock_exec.assert_called_once()
#         with patch.object(qt.QtWidgets.QFileDialog, "getOpenFileName",
#                           return_value=(movie, None)):
#             w.widget.load_video()
#             if movie:
#                 w.widget.save_json()
#                 w.widget.update_tracker()
#     with patch.object(w.widget, "_parse_args", return_value={}):
#         with patch.object(w.widget.tracker, "update_fit_parameters") as update:
#             w.widget.update_tracker()
#             assert(update.call_count == 0)
#     if movie:
#         w.widget.load_video(movie)


# @patch("allensdk.eye_tracking.ui.qt.QtWidgets.QApplication")
# def test_main(mock_app, qtbot, config_file):
#     mock_app.exec_ = MagicMock(return_value=0)
#     args = ["allensdk.eye_tracking_ui"]
#     if config_file:
#         args.extend(["--config_file", config_file])
#     with patch("sys.argv", args):
#         with pytest.raises(SystemExit):
#             __main__.main()


# @patch("allensdk.eye_tracking.ui.qt.QtWidgets.QApplication")
# def test_main_invalid(mock_app, qtbot, movie):
#     mock_app.exec_ = MagicMock(return_value=0)
#     args = ["allensdk.eye_tracking_ui"]
#     if movie:
#         args.extend(["--config_file", movie])
#     with patch("sys.argv", args):
#         with pytest.raises(SystemExit):
#             __main__.main()
