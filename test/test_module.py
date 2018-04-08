from allensdk.eye_tracking import __main__
from allensdk.eye_tracking.frame_stream import CvOutputStream, CvInputStream
import mock
import numpy as np
import ast
import os
import sys
import json
from skimage.draw import circle
import pytest

# H264 is not available by default on windows
if sys.platform == "win32":
    FOURCC = "FMP4"
else:
    FOURCC = "H264"


def image(shape=(200, 200), cr_radius=10, cr_center=(100, 100),
          pupil_radius=30, pupil_center=(100, 100)):
    im = np.ones(shape, dtype=np.uint8)*128
    r, c = circle(pupil_center[0], pupil_center[1], pupil_radius, shape)
    im[r, c] = 0
    r, c = circle(cr_center[0], cr_center[1], cr_radius, shape)
    im[r, c] = 255
    return im


def input_stream(source):
    mock_istream = mock.MagicMock()
    mock_istream.num_frames = 2
    mock_istream.frame_shape = (200, 200)
    mock_istream.__iter__ = mock.MagicMock(
        return_value=iter([np.zeros((200, 200)), np.zeros((200, 200))]))
    return mock_istream


@pytest.fixture()
def input_source(tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("test").join('input.avi'))
    frame = image()
    ostream = CvOutputStream(filename, frame.shape[::-1], is_color=False,
                             fourcc=FOURCC)
    ostream.open(filename)
    for i in range(10):
        ostream.write(frame)
    ostream.close()
    return filename


@pytest.fixture()
def input_json(tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("test").join('input.json'))
    output_dir = str(tmpdir_factory.mktemp("test"))
    annotation_file = str(tmpdir_factory.mktemp("test").join('anno.avi'))
    in_json = {"starburst": {},
               "ransac": {},
               "eye_params": {},
               "qc": {
                   "generate_plots": False,
                   "output_dir": output_dir},
               "annotation": {"annotate_movie": False,
                              "output_file": annotation_file,
                              "fourcc": FOURCC},
               "cr_bounding_box": [],
               "pupil_bounding_box": [],
               "output_dir": output_dir}
    with open(filename, "w") as f:
        json.dump(in_json, f, indent=1)
    return str(filename)


def validate_dict(reference_dict, compare_dict):
    for k, v in reference_dict.items():
        if isinstance(v, dict):
            validate_dict(v, compare_dict[k])
        else:
            assert(compare_dict[k] == v)


def assert_output(output_dir, annotation_file=None, qc_output_dir=None,
                  output_json=None, input_data=None):
    cr = np.load(os.path.join(output_dir, "cr_params.npy"))
    pupil = np.load(os.path.join(output_dir, "pupil_params.npy"))
    assert(os.path.exists(os.path.join(output_dir, "mean_frame.png")))
    assert(cr.shape == (10, 5))
    assert(pupil.shape == (10, 5))
    if annotation_file:
        check = CvInputStream(annotation_file)
        assert(check.num_frames == 10)
        check.close()
    if output_json:
        assert(os.path.exists(output_json))
        if input_data:
            with open(output_json, "r") as f:
                output_data = json.load(f)
                validate_dict(input_data, output_data["input_parameters"])
    if qc_output_dir:
        assert(os.path.exists(os.path.join(output_dir, "cr_all.png")))


def test_main_valid(input_source, input_json, tmpdir_factory):
    output_dir = str(tmpdir_factory.mktemp("output"))
    args = ["allensdk.eye_tracking", "--output_dir", output_dir,
            "--input_source", input_source]
    with mock.patch('sys.argv', args):
        __main__.main()
        assert_output(output_dir)


@pytest.mark.parametrize("pupil_bbox_str,cr_bbox_str, adaptive_pupil", [
    ("[20,50,40,70]", "[40,70,20,50]", True),
    ("[]", "[]", False)
])
def test_main_valid_json(input_source, input_json, pupil_bbox_str, cr_bbox_str,
                    adaptive_pupil):
    args = ["allensdk.eye_tracking", "--input_json", input_json,
            "--input_source", input_source]
    with open(input_json, "r") as f:
        json_data = json.load(f)
    output_dir = json_data["output_dir"]
    with mock.patch('sys.argv', args):
        __main__.main()
        assert_output(output_dir)
    out_json = os.path.join(output_dir, "output.json")
    args.extend(["--qc.generate_plots", "True",
                 "--annotation.annotate_movie", "True",
                 "--output_json", out_json,
                 "--pupil_bounding_box", pupil_bbox_str,
                 "--cr_bounding_box", cr_bbox_str,
                 "--eye_params.adaptive_pupil", str(adaptive_pupil)])
    with mock.patch('sys.argv', args):
        __main__.main()
        json_data["eye_params"]["adaptive_pupil"] = adaptive_pupil
        json_data["qc"]["generate_plots"] = True
        json_data["annotation"]["annotate_movie"] = True
        json_data["output_json"] = out_json
        json_data["pupil_bounding_box"] = ast.literal_eval(pupil_bbox_str)
        json_data["cr_bounding_box"] = ast.literal_eval(cr_bbox_str)
        assert_output(output_dir,
                      annotation_file=json_data["annotation"]["output_file"],
                      qc_output_dir=json_data["qc"]["output_dir"],
                      output_json=out_json,
                      input_data=json_data)
    __main__.plt.close("all")


def test_starburst_override(input_source, input_json):
    args = ["allensdk.eye_tracking", "--input_json", input_json,
            "--input_source", input_source]
    with open(input_json, "r") as f:
        json_data = json.load(f)
    output_dir = json_data["output_dir"]
    out_json = os.path.join(output_dir, "output.json")
    args.extend(["--starburst.cr_threshold_factor", "1.8",
                 "--starburst.pupil_threshold_factor", "2.0",
                 "--starburst.cr_threshold_pixels", "5",
                 "--starburst.pupil_threshold_pixels", "30",
                 "--output_json", out_json])
    with mock.patch('sys.argv', args):
        __main__.main()
        json_data["starburst"]["cr_threshold_factor"] = 1.8
        json_data["starburst"]["pupil_threshold_factor"] = 2.0
        json_data["starburst"]["cr_threshold_pixels"] = 5
        json_data["starburst"]["pupil_threshold_pixels"] = 30
        json_data["output_json"] = out_json
        assert_output(output_dir,
                      output_json=out_json,
                      input_data=json_data)


def test_main_invalid():
    with mock.patch("sys.argv", ["allensdk.eye_tracking"]):
        with mock.patch("argparse.ArgumentParser.print_usage") as mock_print:
            __main__.main()
            mock_print.assert_called_once()
