import os
import numpy as np
import marshmallow
from argschema import ArgSchemaParser
from argschema.utils import schema_argparser
import warnings
import matplotlib
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from ._schemas import InputParameters, OutputParameters  # noqa: E402
from .eye_tracking import EyeTracker  # noqa: E402
from .frame_stream import CvInputStream, CvOutputStream  # noqa: E402
from .plotting import plot_summary, plot_cumulative  # noqa: E402


def setup_annotation(im_shape, annotate_movie, output_file, fourcc="H264"):
    if annotate_movie:
        ostream = CvOutputStream(output_file, im_shape[::-1], fourcc=fourcc)
        ostream.open(output_file)
    else:
        ostream = None
    return ostream


def write_output(output_dir, cr_parameters, pupil_parameters, mean_frame):
    output = {
        "cr_parameter_file": os.path.join(output_dir, "cr_params.npy"),
        "pupil_parameter_file": os.path.join(output_dir, "pupil_params.npy"),
        "mean_frame_file": os.path.join(output_dir, "mean_frame.png")
    }
    plt.imsave(output["mean_frame_file"], mean_frame, cmap="gray")
    np.save(output["cr_parameter_file"], cr_parameters)
    np.save(output["pupil_parameter_file"], pupil_parameters)
    return output


def write_QC_output(output_dir, annotator, cr_parameters, pupil_parameters,
                    mean_frame):
    annotator.annotate_with_cumulative_cr(
        mean_frame, os.path.join(output_dir, "cr_all.png"))
    annotator.annotate_with_cumulative_pupil(
        mean_frame, os.path.join(output_dir, "pupil_all.png"))
    plot_cumulative(annotator.densities["pupil"], annotator.densities["cr"],
                    output_dir=output_dir)
    plot_summary(pupil_parameters, cr_parameters, output_dir=output_dir)


def update_starburst_args(starburst_args):
    threshold_factor = starburst_args.pop("threshold_factor", None)
    threshold_pixels = starburst_args.pop("threshold_pixels", None)

    if starburst_args.get("cr_threshold_factor", None) is None:
        starburst_args["cr_threshold_factor"] = threshold_factor
    if starburst_args.get("pupil_threshold_factor", None) is None:
        starburst_args["pupil_threshold_factor"] = threshold_factor

    if starburst_args.get("cr_threshold_pixels", None) is None:
        starburst_args["cr_threshold_pixels"] = threshold_pixels
    if starburst_args.get("pupil_threshold_pixels", None) is None:
        starburst_args["pupil_threshold_pixels"] = threshold_pixels


def main():
    """Main entry point for running AIBS Eye Tracking."""
    try:
        mod = ArgSchemaParser(schema_type=InputParameters,
                              output_schema_type=OutputParameters)

        update_starburst_args(mod.args["starburst"])

        istream = CvInputStream(mod.args["input_source"])

        im_shape = istream.frame_shape

        ostream = setup_annotation(im_shape, **mod.args["annotation"])

        tracker = EyeTracker(im_shape, istream,
                             ostream,
                             mod.args["starburst"],
                             mod.args["ransac"],
                             mod.args["pupil_bounding_box"],
                             mod.args["cr_bounding_box"],
                             mod.args["qc"]["generate_plots"],
                             **mod.args["eye_params"])
        pupil_parameters, cr_parameters = tracker.process_stream()

        output = write_output(mod.args["output_dir"], cr_parameters,
                              pupil_parameters, tracker.mean_frame)

        if mod.args["qc"]["generate_plots"]:
            write_QC_output(mod.args["qc"]["output_dir"], tracker.annotator,
                            cr_parameters, pupil_parameters,
                            tracker.mean_frame)

        output["input_parameters"] = mod.args
        if "output_json" in mod.args:
            mod.output(output)
        else:
            print(mod.get_output_json(output))
    except marshmallow.ValidationError as e:
        print(e)
        argparser = schema_argparser(InputParameters())
        argparser.print_usage()


if __name__ == "__main__":
    main()
