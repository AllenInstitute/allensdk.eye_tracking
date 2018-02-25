=======
History
=======

1.2.0 (2018-02-24)
------------------
* Rework median filtering and correlation functions to use OpenCV instead of
  scipy for performance improvements.
* Fix seed point finding method to use template matching to improve point finding
  with the default bounding boxes.
* Add keyword arguments to filters for candidate points.
* Expose additional input parameters: average_iris_intensity, clip_pupil_values,
  and max_eccentricity.
* Add constraints to EllipseFitter, preventing any ellipse axis longer than the
  index ray length as well as limiting eccentricity to below max_eccentricity.
* Use the keyword arguments for candidate point filters to filter rays where a
  the baseline intensity is out of pupil limits if clip_pupil_values is set.
* Add plot of average pupil intensity to QC output to check behavior of adaptive
  pupil tracking.
* Add plot of best fit error to QC output.
* Add UI for testing configuration parameters and generating input jsons.

1.1.1 (2018-02-13)
------------------
* Expose median kernel smoothing to the command line.
* Add seed point and candidate pupil points to annotation output.

1.1.0 (2018-02-11)
------------------
* Add frame iteration to allow processing subsets of movies. Also
  add bounding box image to QC output.

1.0.0 (2018-02-07)
------------------
* Rename from aibs.eye_tracking to allensdk.eye_tracking.

0.2.3 (2018-02-06)
------------------
* Add options to set cr_threshold_factor, cr_threshold_pixels, pupil_threshold_factor,
  pupil_threshold_pixels in the starburst parameters. They will override the
  default threshold_factor and threshold_pixels if set.
* Add option to turn off adaptive pupil shade tracking.
  Exposes fourcc string as parameter for annotation in case default codec is not
  supported or desired.

0.2.2 (2018-01-20)
------------------
* Fix matplotlib backend warning.
* Show help if required argument is missing or input command is incorrect.

0.2.1 (2017-12-13)
------------------
* Fix bug preventing module running when number of frames was not specified.

0.2.0 (2017-12-11)
------------------
* Initial release of independent eye tracker.

0.1.0 (2017-10-19)
------------------
* Initial port over of eye tracking code from AllenSDK internal.
