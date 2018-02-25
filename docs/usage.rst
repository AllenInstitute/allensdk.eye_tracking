=====
Usage
=====

After installing the package, and entry point is created so it can be run
from the command line. To minimally run with the default settings::

    allensdk.eye_tracking --input_source <path to an video>

To see all options that can be set at the command line::

    allensdk.eye_tracking --help

There are a lot of options that can be set, so often it can be more
convenient to store them in a JSON-formatted file which can be used like::

    allensdk.eye_tracking --input_json <path to the json>

The input json can be combined with other command line argument, which will
take precedence over anything in the json. There is a UI tool for adjusting
and saving input parameters that can be used by running::

    allensdk.eye_tracking_ui