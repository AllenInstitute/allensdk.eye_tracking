# from allensdk.eye_tracking.ui import qt
# from allensdk.eye_tracking._schemas import InputParameters
# from allensdk.eye_tracking.frame_stream import CvOutputStream
# import os
# import sys
# import numpy as np
# import pytest  # noqa: F401
# from mock import patch

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


# @pytest.fixture(params=[True, False])
# def json_file(tmpdir_factory, request):
#     if request.param:
#         return str(tmpdir_factory.mktemp("test").join("file.json"))
#     else:
#         return ""


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


# @pytest.mark.parametrize("profile", [
#     True, False
# ])
# @patch.object(qt.QtWidgets.QMessageBox, "exec_")
# def test_viewer_window(mock_exec, qtbot, movie, json_file, profile):
#     schema_type = InputParameters
#     w = qt.ViewerWindow(schema_type, profile_runs=profile)
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


# def test_main(qtbot):
#     with patch("sys.argv", ["allensdk.eye_tracking_ui"]):
#         with patch.object(qt.QtWidgets.QApplication, "exec_",
#                           return_value=0):
#             with pytest.raises(SystemExit):
#                 qt.main()
