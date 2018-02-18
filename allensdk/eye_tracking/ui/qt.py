from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure, SubplotParams
import ast
import os
import json
from qtpy import QtCore, QtWidgets
from argschema.schemas import mm
from argschema import ArgSchemaParser
from allensdk.eye_tracking import _schemas
from allensdk.eye_tracking.frame_stream import CvInputStream
from allensdk.eye_tracking.eye_tracking import EyeTracker
from allensdk.eye_tracking.__main__ import get_starburst_args
from allensdk.eye_tracking.plotting import annotate_with_box


LITERAL_EVAL_TYPES = {_schemas.NumpyArray, _schemas.Bool}


class FieldWidget(QtWidgets.QLineEdit):
    def __init__(self, key, field, parent=None):
        if field.default == mm.missing:
            default = ""
        else:
            default = str(field.default)
        super(FieldWidget, self).__init__(default, parent=parent)
        self.field = field
        self.key = key

    def get_json(self):
        raw_value = str(self.text())
        if raw_value:
            if type(self.field) in LITERAL_EVAL_TYPES:
                raw_value = ast.literal_eval(raw_value)
            value = self.field.deserialize(raw_value)
            if isinstance(self.field, _schemas.NumpyArray):
                value = value.tolist()
            return value
        return None


class SchemaWidget(QtWidgets.QWidget):
    def __init__(self, key, schema, parent=None):
        super(SchemaWidget, self).__init__(parent=parent)
        self.key = key
        self.schema = schema
        self.fields = {}
        self.layout = QtWidgets.QGridLayout()
        self._init_widgets()
        self.setLayout(self.layout)

    def _init_widgets(self):
        fields = {}
        nested = {}
        for k, v in self.schema.fields.items():
            if isinstance(v, _schemas.Nested):
                w = SchemaWidget(k, v.schema, self)
                nested[k] = w
            else:
                w = FieldWidget(k, v, self)
                fields[k] = w
            self.fields[k] = w
        self._init_layout(fields, nested)

    def _init_layout(self, fields, nested):
        i = 0
        if self.key is not None:
            label = QtWidgets.QLabel("<b>{}</b>".format(self.key))
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.layout.addWidget(label, i, 0, 1, 2)
            i += 1
        for k, v in sorted(fields.items()):
            label = QtWidgets.QLabel("{}: ".format(k))
            self.layout.addWidget(label, i, 0)
            self.layout.addWidget(v, i, 1)
            i += 1
        for k, v in sorted(nested.items()):
            self.layout.addWidget(v, i, 0, 1, 2)
            i += 1

    def get_json(self):
        json_data = {}
        for k, v in self.fields.items():
            data = v.get_json()
            if data is not None:
                json_data[k] = data
        if json_data:
            return json_data
        return None


class InputJsonWidget(QtWidgets.QScrollArea):
    def __init__(self, schema, parent=None):
        super(InputJsonWidget, self).__init__(parent=parent)
        self.setWindowTitle("Input Parameters")
        self.schema_widget = SchemaWidget(None, schema, self)
        self.setWidget(self.schema_widget)

    def get_json(self):
        return self.schema_widget.get_json()


class ViewerWidget(QtWidgets.QWidget):
    def __init__(self, schema_type):
        super(ViewerWidget, self).__init__()
        self.layout = QtWidgets.QGridLayout()
        self.schema_type = schema_type
        self.video = "./"
        self._init_widgets()
        self.tracker = self._setup_tracker()
        self.setLayout(self.layout)

    def _init_widgets(self):
        sp_params = SubplotParams(0, 0, 1, 1)
        self.figure = Figure(frameon=False, subplotpars=sp_params)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.json_view = InputJsonWidget(self.schema_type(), parent=self)
        self.rerun_button = QtWidgets.QPushButton("Reprocess Frame",
                                                  parent=self)
        self.save_button = QtWidgets.QPushButton("Save Json", parent=self)
        self.slider = QtWidgets.QSlider(parent=self)
        self.slider.setMinimum(0)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self._connect_signals()
        self._init_layout()

    def _init_layout(self):
        self.layout.addWidget(self.canvas, 0, 0)
        self.layout.addWidget(self.json_view, 0, 1)
        self.layout.addWidget(self.slider, 1, 0)
        self.layout.addWidget(self.rerun_button, 2, 0)
        self.layout.addWidget(self.save_button, 2, 1)

    def _connect_signals(self):
        self.slider.sliderReleased.connect(self.show_frame)
        self.rerun_button.clicked.connect(self.update_tracker)
        self.save_button.clicked.connect(self.save_json)

    def _parse_args(self, json_data):
        try:
            mod = ArgSchemaParser(input_data=json_data,
                                  schema_type=self.schema_type)
            return mod.args
        except Exception as e:
            print(e)
        return None

    def _setup_tracker(self):
        json_data = self.json_view.get_json()
        json_data["input_source"] = os.path.abspath(__file__)
        args = self._parse_args(json_data)
        if args:
            tracker = EyeTracker(
                None, None, get_starburst_args(args["starburst"]),
                args["ransac"], args["pupil_bounding_box"],
                args["cr_bounding_box"], False, **args["eye_params"])
        else:
            tracker = None
        return tracker

    def update_tracker(self):
        json_data = self.json_view.get_json()
        input_source = os.path.normpath(json_data.get("input_source", "./"))
        load = False
        if not os.path.isfile(input_source):
            json_data["input_source"] = os.path.abspath(__file__)
        elif self.video != input_source:
            load = True
        args = self._parse_args(json_data)
        if args:
            self.tracker.update_fit_parameters(
                get_starburst_args(args["starburst"]), args["ransac"],
                args["pupil_bounding_box"], args["cr_bounding_box"],
                **args["eye_params"])
        if load:
            self.load_video(input_source)
        else:
            self.show_frame()

    def save_json(self):
        json_data = self.json_view.get_json()
        valid = self._parse_args(json_data)
        if valid:
            filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Json file")
            if filepath:
                with open(filepath, "w") as f:
                    json.dump(json_data, f, indent=1)

    def load_video(self, path):
        self.video = os.path.normpath(path)
        input_stream = CvInputStream(self.video)
        self.tracker.input_stream = input_stream
        self.slider.setMaximum(input_stream.num_frames-1)
        self.slider.setValue(0)
        self.show_frame()

    def show_frame(self):
        ax = self.figure.add_subplot(111)
        ax.clear()
        frame = self.tracker.input_stream[self.slider.value()]
        cr, pupil = self.tracker.process_image(frame)
        anno = self.tracker.annotator.annotate_frame(
            self.tracker.current_image, pupil, cr, self.tracker.current_seed,
            self.tracker.current_pupil_candidates)
        self.tracker.annotator.clear_rc()
        anno = annotate_with_box(anno, self.tracker.cr_bounding_box,
                                 (0, 0, 255))
        anno = annotate_with_box(anno, self.tracker.pupil_bounding_box,
                                 (255, 0, 0))
        ax.imshow(anno[:, :, ::-1], interpolation="none")
        ax.axis("off")
        self.canvas.draw()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication([])
    w = ViewerWidget(_schemas.InputParameters)
    w.show()
    sys.exit(app.exec_())
