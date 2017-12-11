from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (Nested, OutputDir, InputFile, Bool, Float, Int,
                              OutputFile, NumpyArray)


class RansacParameters(DefaultSchema):
    minimum_points_for_fit = Int(default=10,
                                 description=("Number of points required to "
                                              "fit data"))
    number_of_close_points = Int(default=4,
                                 description=("Number of candidate outliers "
                                              "reselected as inliers required "
                                              "to consider a good fit"))
    threshold = Float(default=0.0001,
                      description=("Error threshold below which data should "
                                   "be considered an inlier"))
    iterations = Int(default=10,
                     description="Number of iterations to run")


class AnnotationParameters(DefaultSchema):
    annotate_movie = Bool(default=False,
                          description="Flag for whether or not to annotate")
    output_file = OutputFile(default="./annotated.avi")


class StarburstParameters(DefaultSchema):
    index_length = Int(default=200,
                       description="Initial default length for rays")
    n_rays = Int(default=18,
                 description="Number of rays to draw")
    threshold_factor = Float(default=1.6,
                             description="Threshold factor for ellipse edges")
    threshold_pixels = Int(default=10,
                           description=("Number of pixels from start of ray "
                                        "to use for adaptive threshold, "
                                        "also serves as a minimum cutoff for "
                                        "point detection"))


class EyeParameters(DefaultSchema):
    cr_recolor_scale_factor = Float(default=2.0,
                                    description=("Size multiplier for corneal "
                                                 "reflection recolor mask"))
    min_pupil_value = Int(default=0,
                          description=("Minimum value the average pupil shade "
                                       "can be"))
    max_pupil_value = Int(default=30,
                          description=("Maximum value the average pupil shade "
                                       "can be"))
    recolor_cr = Bool(default=True,
                      description="Flag for recoloring corneal reflection")
    pupil_mask_radius = Int(default=40,
                            description=("Radius of pupil mask used to find "
                                         "seed point"))
    cr_mask_radius = Int(default=10,
                         description=("Radius of cr mask used to find seed "
                                     "point"))


class QCParameters(DefaultSchema):
    generate_plots = Bool(default=False,
                          description=("Flag for whether or not to output QC "
                                       "plots"))
    output_dir = OutputDir(default="./qc",
                           description="Folder to store QC outputs")


class InputParameters(ArgSchema):
    # Add your input parameters
    output_dir = OutputDir(default="./",
                           description=("Directory in which to store data "
                                        "output files"))
    input_source = InputFile(description="Path to input movie",
                             required=True)
    pupil_bounding_box = NumpyArray(dtype="int",
                                    default=None)
    cr_bounding_box = NumpyArray(dtype="int",
                                 default=None)
    ransac = Nested(RansacParameters)
    annotation = Nested(AnnotationParameters)
    starburst = Nested(StarburstParameters)
    eye_params = Nested(EyeParameters)
    qc = Nested(QCParameters)


class OutputSchema(DefaultSchema):
    input_parameters = Nested(InputParameters,
                              description=("Input parameters the module was "
                                           "run with"),
                              required=True)


class OutputParameters(OutputSchema):
    cr_parameter_file = OutputFile(required=True)
    pupil_parameter_file = OutputFile(required=True)
    mean_frame_file = OutputFile(required=True)
