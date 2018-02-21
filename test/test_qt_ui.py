from allensdk.eye_tracking.ui import qt
from allensdk.eye_tracking._schemas import InputParameters
import os
import pytest
from mock import patch


def test_field_widget(qtbot):
    schema = InputParameters()
    w = qt.FieldWidget("test", schema.fields["output_dir"])
    assert(w.key == "test")
    assert(w.field == schema.fields["output_dir"])
    js = w.get_json()
    default = schema.fields["output_dir"].default
    assert(os.path.normpath(js) == os.path.normpath(default))

    w = qt.FieldWidget("test2", schema.fields["input_source"])
    assert(w.key == "test2")
    assert(str(w.text()) == "")
    js = w.get_json()
    assert(js is None)

    w = qt.FieldWidget("test3", schema.fields["pupil_bounding_box"])
    assert(w.key == "test3")
    default = schema.fields["pupil_bounding_box"].default
    assert(str(w.text()) == str(default))
    js = w.get_json()
    assert(js == default)


def test_schema_widget(qtbot):
    schema = InputParameters()
    w = qt.SchemaWidget(None, schema, None)
    assert(w.key is None)
    assert(isinstance(w.fields["eye_params"], qt.SchemaWidget))
    assert(isinstance(w.fields["input_source"], qt.FieldWidget))

    js = w.get_json()
    assert(js)
    with patch.object(qt.FieldWidget, "get_json", return_value=None):
        js = w.get_json()
        assert(js is None)

    w.update_value("eye_params.min_pupil_value", "1000")
    val = str(w.fields["eye_params"].fields["min_pupil_value"].text())
    assert(val == "1000")
