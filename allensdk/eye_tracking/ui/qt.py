from qtpy import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure, SubplotParams
import ast
import os
import json
import logging
import cProfile
from argschema.schemas import mm
from argschema import ArgSchemaParser
from allensdk.eye_tracking import _schemas
from allensdk.eye_tracking.frame_stream import CvInputStream
from allensdk.eye_tracking.eye_tracking import EyeTracker
from allensdk.eye_tracking.__main__ import get_starburst_args
from allensdk.eye_tracking.plotting import annotate_with_box


LITERAL_EVAL_TYPES = {_schemas.NumpyArray, _schemas.Bool}


class FieldWidget(QtWidgets.QLineEdit):
    """Widget for displaying and editing a schema field.

    Parameters
    ----------
    key : string
        Name of the field.
    field : argschema.Field
        Argschema Field object containing serialization and default
        data information.
    parent : QtWidgets.QWidget
        Parent widget.
    """
    def __init__(self, key, field, parent=None, **kwargs):
        if field.default == mm.missing:
            default = ""
        else:
            default = str(field.default)
        super(FieldWidget, self).__init__(default, parent=parent)
        self.field = field
        self.key = key
        self.setEnabled(not kwargs.get("read_only", False))
        self.displayed = kwargs.get("visible", True)
        self.setVisible(self.displayed)

    def get_json(self):
        """Get the JSON serializable data from this field.

        Returns
        -------
        data : object
            JSON serializable data in the widget, or None if empty.
        """
        raw_value = str(self.text())
        if raw_value:
            if type(self.field) in LITERAL_EVAL_TYPES:
                try:
                    raw_value = ast.literal_eval(raw_value)
                except SyntaxError:
                    pass  # let validation handle it
            value = self.field.deserialize(raw_value)
            if isinstance(self.field, _schemas.NumpyArray):
                value = value.tolist()
            return value
        return None


class SchemaWidget(QtWidgets.QWidget):
    """Widget for displaying an ArgSchema.

    Parameters
    ----------
    key : string
        The key of the schema if it is nested.
    schema : argschema.DefaultSchema
        The schema to create a widget for.
    parent : QtWidgets.QWidget
        Parent widget.
    """
    def __init__(self, key, schema, parent=None, config=None):
        super(SchemaWidget, self).__init__(parent=parent)
        self.key = key
        self.schema = schema
        self.fields = {}
        self.config = config
        if config is None:
            self.config = {}
        self.layout = QtWidgets.QGridLayout()
        all_children_hidden = self._init_widgets()
        self.setLayout(self.layout)
        self.displayed = not all_children_hidden
        self.setVisible(self.displayed)

    def _init_widgets(self):
        fields = {}
        nested = {}
        all_hidden = True
        for k, v in self.schema.fields.items():
            if isinstance(v, _schemas.Nested):
                w = SchemaWidget(k, v.schema, self,
                                 config=self.config.get(k, {}))
                nested[k] = w
            else:
                w = FieldWidget(k, v, self, **self.config.get(k, {}))
                fields[k] = w
            self.fields[k] = w
            if w.displayed:
                all_hidden = False
        self._init_layout(fields, nested)
        return all_hidden

    def _init_layout(self, fields, nested):
        i = 0
        if self.key is not None:
            label = QtWidgets.QLabel("<b>{}</b>".format(self.key))
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.layout.addWidget(label, i, 0, 1, 2)
            i += 1
        for k, v in sorted(fields.items()):
            label = QtWidgets.QLabel("{}: ".format(k))
            label.setVisible(v.displayed)
            self.layout.addWidget(label, i, 0)
            self.layout.addWidget(v, i, 1)
            i += 1
        for k, v in sorted(nested.items()):
            self.layout.addWidget(v, i, 0, 1, 2)
            i += 1

    def get_json(self):
        """Get the JSON serializable data from this schema.

        Returns
        -------
        data : object
            JSON serializable data in the widget, or None if empty.
        """
        json_data = {}
        for key, value in self.fields.items():
            data = value.get_json()
            if data is not None:
                json_data[key] = data
        if json_data:
            return json_data
        return None

    def update_value(self, attribute, value):
        """Update a value in the schema.

        Parameters
        ----------
        attribute : string
            Attribute name to update.
        value : string
            Value to set the field edit box to.
        """
        attrs = attribute.split(".", 1)
        if len(attrs) > 1:
            self.fields[attrs[0]].update_value(attrs[1], value)
        else:
            self.fields[attribute].setText(value)


class InputJsonWidget(QtWidgets.QScrollArea):
    """Widget for displaying an editable input json in a scroll area.

    Parameters
    ----------
    schema : argschema.DefaultSchema
        Schema from which to build widgets.
    parent : QtWidgets.QWidget
        Parent widget.
    """
    def __init__(self, schema, parent=None, config=None):
        super(InputJsonWidget, self).__init__(parent=parent)
        self.schema_widget = SchemaWidget(None, schema, self, config)
        self.setWidget(self.schema_widget)

    def get_json(self):
        return self.schema_widget.get_json()

    def update_value(self, attribute, value):
        self.schema_widget.update_value(attribute, value)


class BBoxCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas widget with drawable box.

    Parameters
    ----------
    figure : matplotlib.Figure
        Matplob figure to contain in the canvas.
    """
    box_updated = QtCore.Signal(int, int, int, int)

    def __init__(self, figure):
        super(BBoxCanvas, self).__init__(figure)
        self.rgba = (255, 255, 255, 20)
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.drawing = False

    def set_rgb(self, r, g, b):
        """Set the RGB values for the bounding box tool.

        Parameters
        ----------
        r : int
            Red channel value (0-255).
        g : int
            Green channel value (0-255).
        b : int
            Blue channel value (0-255).
        """
        self.rgba = (r, g, b, 20)

    def paintEvent(self, event):
        """Event override for painting to draw bounding box.

        Parameters
        ----------
        event : QtCore.QEvent
            The paint event.
        """
        super(BBoxCanvas, self).paintEvent(event)
        if self.drawing:
            painter = QtGui.QPainter(self)
            brush = QtGui.QBrush(QtGui.QColor(*self.rgba))
            painter.setBrush(brush)
            painter.drawRect(QtCore.QRect(self.begin, self.end))

    def mousePressEvent(self, event):
        """Event override for painting to initialize bounding box.

        Parameters
        ----------
        event : QtCore.QEvent
            The mouse press event.
        """
        self.begin = event.pos()
        self.end = event.pos()
        self.drawing = True
        self.update()

    def mouseMoveEvent(self, event):
        """Event override for painting to update bounding box.

        Parameters
        ----------
        event : QtCore.QEvent
            The mouse move event.
        """
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        """Event override for painting to finalize bounding box.

        Parameters
        ----------
        event : QtCore.QEvent
            The mouse release event.
        """
        self.end = event.pos()
        self.update()
        self.drawing = False
        x1 = self.begin.x()
        x2 = self.end.x()
        y1 = self.begin.y()
        y2 = self.end.y()
        self.box_updated.emit(min(x1, x2), max(x1, x2),
                              min(y1, y2), max(y1, y2))


class ViewerWidget(QtWidgets.QWidget):
    """Widget for tweaking eye tracking parameters and viewing output.

    Parameters
    ----------
    schema_type : type(argschema.DefaultSchema)
        The input schema type.
    """
    def __init__(self, schema_type, profile_runs=False, parent=None,
                 config=None):
        super(ViewerWidget, self).__init__(parent=parent)
        self.profile_runs = profile_runs
        self.layout = QtWidgets.QGridLayout()
        self.schema_type = schema_type
        self.video = "./"
        self._init_widgets(config)
        self.tracker = EyeTracker(None, None)
        self.update_tracker()
        self.setLayout(self.layout)

    def _init_widgets(self, config):
        sp_params = SubplotParams(0, 0, 1, 1)
        self.figure = Figure(frameon=False, subplotpars=sp_params)
        self.axes = self.figure.add_subplot(111)
        self.canvas = BBoxCanvas(self.figure)
        self.json_view = InputJsonWidget(self.schema_type(), parent=self,
                                         config=config)
        self.rerun_button = QtWidgets.QPushButton("Reprocess Frame",
                                                  parent=self)
        self.pupil_radio = QtWidgets.QRadioButton("Pupil BBox", parent=self)
        self.cr_radio = QtWidgets.QRadioButton("CR BBox", parent=self)
        self.slider = QtWidgets.QSlider(parent=self)
        self.slider.setMinimum(0)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self._connect_signals()
        self._init_layout()

    def _init_layout(self):
        self.layout.addWidget(self.canvas, 0, 0, 1, 2)
        self.layout.addWidget(self.json_view, 0, 2, 1, 2)
        self.layout.addWidget(self.slider, 1, 0, 1, 2)
        self.layout.addWidget(self.rerun_button, 2, 0)
        self.layout.addWidget(self.pupil_radio, 2, 1)
        self.layout.addWidget(self.cr_radio, 2, 2)

    def _connect_signals(self):
        self.slider.sliderReleased.connect(self.show_frame)
        self.rerun_button.clicked.connect(self.update_tracker)
        self.canvas.box_updated.connect(self.update_bbox)
        self.pupil_radio.clicked.connect(self._setup_bbox)
        self.cr_radio.clicked.connect(self._setup_bbox)

    def _setup_bbox(self):
        if self.pupil_radio.isChecked():
            self.canvas.set_rgb(0, 0, 255)
        elif self.cr_radio.isChecked():
            self.canvas.set_rgb(255, 0, 0)

    def update_bbox(self, xmin, xmax, ymin, ymax):
        bbox = [xmin, xmax, ymin, ymax]
        if self.pupil_radio.isChecked():
            self.json_view.update_value("pupil_bounding_box", str(bbox))
            self.update_tracker()
        elif self.cr_radio.isChecked():
            self.json_view.update_value("cr_bounding_box", str(bbox))
            self.update_tracker()

    def _parse_args(self, json_data):
        try:
            mod = ArgSchemaParser(input_data=json_data,
                                  schema_type=self.schema_type)
            return mod.args
        except Exception as e:
            self._json_error_popup(e)
        return None

    def get_json_data(self):
        try:
            return self.json_view.get_json()
        except Exception as e:
            self._json_error_popup(e)

    def update_tracker(self):
        json_data = self.get_json_data()
        if json_data is None:
            return
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
            self._load_video(input_source)
        elif self.tracker.input_stream is not None:
            self.show_frame()

    def save_json(self):
        json_data = self.get_json_data()
        if json_data is None:
            return
        valid = self._parse_args(json_data)
        if valid:
            filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Json file")
            if os.path.exists(os.path.dirname(filepath)):
                with open(filepath, "w") as f:
                    json.dump(json_data, f, indent=1)

    def load_video(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video")
        filepath = filepath.strip("'\" ")
        if os.path.exists(filepath):
            self.json_view.update_value("input_source",
                                        os.path.normpath(filepath))
            self.update_tracker()

    def _load_video(self, path):
        self.video = os.path.normpath(path)
        input_stream = CvInputStream(self.video)
        self.tracker.input_stream = input_stream
        self.slider.setMaximum(input_stream.num_frames-1)
        self.slider.setValue(0)
        self.show_frame()

    def show_frame(self, n=None):
        self.axes.clear()
        frame = self.tracker.input_stream[self.slider.value()]
        if self.profile_runs:
            p = cProfile.Profile()
            p.enable()
        self.tracker.last_pupil_color = self.tracker.min_pupil_value
        cr, pupil, cr_err, pupil_err = self.tracker.process_image(frame)
        anno = self.tracker.annotator.annotate_frame(
            self.tracker.current_image, pupil, cr, self.tracker.current_seed,
            self.tracker.current_pupil_candidates)
        self.tracker.annotator.clear_rc()
        if self.profile_runs:
            p.disable()
            p.print_stats('cumulative')
        anno = annotate_with_box(anno, self.tracker.cr_bounding_box,
                                 (0, 0, 255))
        anno = annotate_with_box(anno, self.tracker.pupil_bounding_box,
                                 (255, 0, 0))
        self.axes.imshow(anno[:, :, ::-1], interpolation="none")
        self.axes.axis("off")
        self.canvas.draw()

    def _json_error_popup(self, msg):
        message = "<pre>Error parsing input json: \n{}</pre>".format(msg)
        box = QtWidgets.QMessageBox(self)
        box.setText(message)
        box.exec_()


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, schema_type, profile_runs=False, config=None):
        super(ViewerWindow, self).__init__()
        self.setWindowTitle("Eye Tracking Configuration Tool")
        self.widget = ViewerWidget(schema_type, profile_runs=profile_runs,
                                   parent=self, config=config)
        self.setCentralWidget(self.widget)
        self._init_menubar()

    def _init_menubar(self):
        file_menu = self.menuBar().addMenu("&File")
        load = file_menu.addAction("Load Video")
        save = file_menu.addAction("Save JSON")
        load.triggered.connect(self.widget.load_video)
        save.triggered.connect(self.widget.save_json)


def load_config(config_file):
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except Exception as e:
        logging.error(e)
        config = None
    return config


def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--config_file", type=str, default="")
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left
    config = None
    if args.config_file:
        config = load_config(args.config_file)
    app = QtWidgets.QApplication([])
    w = ViewerWindow(_schemas.InputParameters, args.profile, config)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
