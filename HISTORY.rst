=======
History
=======

1.0.0 (2018-02-07)
------------------
Rename from aibs.eye_tracking to allensdk.eye_tracking.

0.2.3 (2018-02-06)
------------------
Add options to set cr_threshold_factor, cr_threshold_pixels, pupil_threshold_factor,
pupil_threshold_pixels in the starburst parameters. They will override the
default threshold_factor and threshold_pixels if set.
Add option to turn off adaptive pupil shade tracking.
Exposes fourcc string as parameter for annotation in case default codec is not
supported or desired.

0.2.2 (2018-01-20)
------------------
Fix matplotlib backend warning.
Show help if required argument is missing or input command is incorrect.

0.2.1 (2017-12-13)
------------------
Fix bug preventing module running when number of frames was not specified.

0.2.0 (2017-12-11)
------------------
Initial release of independent eye tracker.

0.1.0 (2017-10-19)
------------------
Initial port over of eye tracking code from AllenSDK internal.
