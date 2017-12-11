import os
import numpy as np
from matplotlib import pyplot as plt
from ._schemas import InputParameters, OutputParameters
from argschema import ArgSchemaParser
from .eye_tracking import EyeTracker
from .frame_stream import CvInputStream, CvOutputStream
from .plotting import plot_summary, plot_cumulative


def setup_annotation(im_shape, annotate_movie, output_file):
    if annotate_movie:
        ostream = CvOutputStream(output_file, im_shape[::-1])
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


def main():
    """Main entry point for running AIBS Eye Tracking."""
    mod = ArgSchemaParser(schema_type=InputParameters,
                          output_schema_type=OutputParameters)
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
                        cr_parameters, pupil_parameters, tracker.mean_frame)

    output["input_parameters"] = mod.args
    if "output_json" in mod.args:
        mod.output(output)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":
    main()
