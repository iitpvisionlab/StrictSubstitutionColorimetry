# StrictSubstitutionColorimetry

This repository is a set of software for visualisation and analysis of data described in a paper: "The first use of the strict substitution colorimetry for collecting
data on threshold perceptual color differences in humans".

Spectral power distribution of the Red, Green, and Blue channels of the stimulus generator you can find in StimulusGenerator-SPD.csv

## Benchmarking

To run the benchmarking software, use the following command:

`python -m benchmarking [-h] [--no-group] [--centers CENTERS [CENTERS ...]] [--metric {stress,cv,pf/3}] dataset`

Usage example:

`python -m benchmarking data\participants\Participant_1.json`

As the dataset, you may either use combvd, munsell, sdcth or specify a list of files in the XLSX, ODS or JSON format, which contain the measurement data presented in the specific form accepted in this project.

The `--metric` option is used to set the metric used to compare the dataset and the model. It can take such values as stress or pf/3.

For the results of measurements executed in the current project, the following extra options are available:

1. The `--no-group` key is used to calculate the standard STRESS instead of the GroupSTRESS.

2. The centers option allows one to enumerate a list of color centers separated by a space, the measurements in which must be taken into account when calculating the statistics. By defaul all the centers in measurement data are used in calculations.

## Observer variablility

To run the intra- and inter-observer variablility calculating software, use the following command:

`python -m obs_variablility`

## Visualization

To run the visualization software, use the following command:

`python -m visualization [-h] [--plane] [--try]`

Usage example:

`python -m visualization data\participants\Participant_1.json`

The `--plane` option is used to set a plane of visualisation: `xy`, `xY`, `yY` or `3D` for 3D viasualisation

The `--try` option is used to show all attempts of color differences evaluation during a session

## Data description

Measured data consist of 4 datasets corresponding for each participant. Each dataset is a JSON-file describing measurement
results.

The keys center_x, center_y and center_Y specify the coordinates of each color center around which measurements were taken. The coordinates are specified in the xyY color coordinate system in the color space of the standard observer CIE 1964 (10-degree standart observer). The key Yc is a coefficient representing individual scale ratio between luminocity axis Y and chromaticity plane xy.

For each color center for each of 18 directions for each of 4 attempts a set of keys x, y, Y, angle_1, angle_2 is specified. Here x, y, Y are the coordinates of the threshold point, shifted from the color center in the direction (angle_1, angle_2), which are set respectively from the x axis and xy chromaticity plane. Angles are specified in degrees. The angle_2 value of ±45 degrees is interpreted as a combination of luminance and chroma shifts.
