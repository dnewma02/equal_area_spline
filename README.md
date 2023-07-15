# ea_spline.py

A python implemetation of ea_spline.R from the `ithir` package.

Fits a mass-preserving (pycnophylactic) spline to soil profile data. See [Bishop et al. (1999)](http://dx.doi.org/10.1016/S0016-7061(99)00003-8) for details.

Hosted here until a python based digital soil mapping module is assembled.

## Requirements

This package requires `python`, `numpy` and `pandas`, versions 3.6, 1.24, and 2.0 respectively.

>python -m pip install numpy pandas

## Usage

Can be used byimporting `ea_spline()` or using command line:

>python ea_spline.py --data=path/to/data.csv --var=VariableHeading --output=path/to/write/output.csv

**Note:** the input data must be structured such that the first column is the site identifier, the second column is the upper limit of the horizon (closest to the surface), and the third column is the lower limit of the horizon (furthest from the surface).

### Parameters

The following parameters can be combined with the '--' flag in command line.

| Parameter | Required | Type | Defualt | Description |
| --------- | -------- | ---- | ------- | ----------- |
| data | True | str | None | Path to the input .csv data |
| var | True | str | None | Variable column name as it appears in the .csv header |
| output | True | str | None | Path to write an output file |
| depths | False | list(integers) | 0,5,15,30,60,100,200 | Comma separated depth interval integers in cm. Defaults to GlobalSoilMap depths |
| lam | False | float | 0.1 | Equal area spline lambda parameter |
| vhigh | False | int | 1000 | Sets the maximum of the fitted variable range |
| vlow | False | int | 0 | Sets the minimum of the fitted variable range |
| show_prog | False | bool | False | Set to True to print progress updates to the standard out |
