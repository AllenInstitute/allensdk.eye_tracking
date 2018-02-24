from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (Nested, OutputDir, InputFile, Bool, Float, Int,
                              OutputFile, NumpyArray, Str)
from .eye_tracking import PointGenerator, EyeTracker
from .fit_ellipse import EllipseFitter


class RansacParameters(DefaultSchema):
    minimum_points_for_fit = Int(
        default=EllipseFitter.DEFAULT_MINIMUM_POINTS_FOR_FIT,
        description="Number of points required to fit data")
    number_of_close_points = Int(
        default=EllipseFitter.DEFAULT_NUMBER_OF_CLOSE_POINTS,
        description=("Number of candidate outliers reselected as inliers "
                     "required to consider a good fit"))
    threshold = Float(
        default=EllipseFitter.DEFAULT_THRESHOLD,
        description=("Error threshold below which data should be considered "
                     "an inlier"))
    iterations = Int(
        default=EllipseFitter.DEFAULT_ITERATIONS,
        description="Number of iterations to run")


class AnnotationParameters(DefaultSchema):
    annotate_movie = Bool(
        default=False,
        description="Flag for whether or not to annotate")
    output_file = OutputFile(default="./annotated.avi")
    fourcc = Str(description=("FOURCC string for video encoding. On Windows "
                              "H264 is not available by default, so it will "
                              "need to be installed or a different codec "
                              "used."))


class StarburstParameters(DefaultSchema):
    index_length = Int(
        default=PointGenerator.DEFAULT_INDEX_LENGTH,
        description="Initial default length for rays")
    n_rays = Int(
        default=PointGenerator.DEFAULT_N_RAYS,
        description="Number of rays to draw")
    threshold_factor = Float(
        default=PointGenerator.DEFAULT_THRESHOLD_FACTOR,
        description="Threshold factor for ellipse edges")
    cr_threshold_factor = Float(
        description=("Threshold factor for corneal reflection ellipse edges, "
                     "will supercede `threshold_factor` for corneal "
                     "reflection if specified"))
    pupil_threshold_factor = Float(
        description=("Threshold factor for pupil ellipse edges, will "
                     "supercede `threshold_factor` for pupil if specified"))
    threshold_pixels = Int(
        default=PointGenerator.DEFAULT_THRESHOLD_PIXELS,
        description=("Number of pixels from start of ray to use for adaptive "
                     "threshold, also serves as a minimum cutoff for point "
                     "detection"))
    cr_threshold_pixels = Int(
        description=("Number of pixels from start of ray to use for adaptive "
                     "threshold of the corneal reflection, will supercede "
                     "`threshold_pixels` for corneal reflection if specified. "
                     "Also serves as a minimum cutoff for point detection"))
    pupil_threshold_pixels = Int(
        description=("Number of pixels from start of ray to use for adaptive "
                     "threshold of the pupil, will supercede `threshold_pixels"
                     "` for pupil if specified. Also serves as a minimum "
                     "cutoff for point detection"))


class EyeParameters(DefaultSchema):
    cr_recolor_scale_factor = Float(
        default=EyeTracker.DEFAULT_CR_RECOLOR_SCALE_FACTOR,
        description="Size multiplier for corneal reflection recolor mask")
    min_pupil_value = Int(
        default=EyeTracker.DEFAULT_MIN_PUPIL_VALUE,
        description="Minimum value the average pupil shade can be")
    max_pupil_value = Int(
        default=EyeTracker.DEFAULT_MAX_PUPIL_VALUE,
        description="Maximum value the average pupil shade can be")
    recolor_cr = Bool(
        default=EyeTracker.DEFAULT_RECOLOR_CR,
        description="Flag for recoloring corneal reflection")
    adaptive_pupil = Bool(
        default=EyeTracker.DEFAULT_ADAPTIVE_PUPIL,
        description="Flag for whether or not to adaptively update pupil color")
    pupil_mask_radius = Int(
        default=EyeTracker.DEFAULT_PUPIL_MASK_RADIUS,
        description="Radius of pupil mask used to find seed point")
    cr_mask_radius = Int(
        default=EyeTracker.DEFAULT_CR_MASK_RADIUS,
        description="Radius of cr mask used to find seed point")
    smoothing_kernel_size = Int(
        default=EyeTracker.DEFAULT_SMOOTHING_KERNEL_SIZE,
        description=("Kernel size for median filter smoothing kernel (must be "
                     "odd)"))
    clip_pupil_values = Bool(
        default=EyeTracker.DEFAULT_CLIP_PUPIL_VALUES,
        description=("Flag of whether or not to restrict pupil values for "
                     "starburst to fall within the range of (min_pupil_value, "
                     "max_pupil_value)"))
    average_iris_intensity = Int(
        default=EyeTracker.DEFAULT_AVERAGE_IRIS_INTENSITY,
        description="Average expected intensity of the iris")
    max_eccentricity = Float(
        default=EyeTracker.DEFAULT_MAX_ECCENTRICITY,
        description="Maximum eccentricity allowed for pupil.")


class QCParameters(DefaultSchema):
    generate_plots = Bool(
        default=EyeTracker.DEFAULT_GENERATE_QC_OUTPUT,
        description="Flag for whether or not to output QC plots")
    output_dir = OutputDir(
        default="./qc",
        description="Folder to store QC outputs")


class InputParameters(ArgSchema):
    output_dir = OutputDir(
        default="./",
        description="Directory in which to store data output files")
    input_source = InputFile(
        description="Path to input movie",
        required=True)
    pupil_bounding_box = NumpyArray(dtype="int", default=[])
    cr_bounding_box = NumpyArray(dtype="int", default=[])
    start_frame = Int(
        description="Frame of movie to start processing at")
    stop_frame = Int(
        description="Frame of movie to end processing at")
    frame_step = Int(
        description=("Interval of frames to process. Used for skipping frames,"
                     "if 1 it will process every frame between start and stop")
    )
    ransac = Nested(RansacParameters)
    annotation = Nested(AnnotationParameters)
    starburst = Nested(StarburstParameters)
    eye_params = Nested(EyeParameters)
    qc = Nested(QCParameters)


class OutputSchema(DefaultSchema):
    input_parameters = Nested(
        InputParameters,
        description="Input parameters the module was run with",
        required=True)


class OutputParameters(OutputSchema):
    cr_parameter_file = OutputFile(required=True)
    pupil_parameter_file = OutputFile(required=True)
    mean_frame_file = OutputFile(required=True)
