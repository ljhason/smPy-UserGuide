# smPy

## The Development of a Global Software Analysis Tool for Single-molecule Imaging

The importance of single-molecule microscopy is motivated by the useful information that can be gleaned by observing the dynamic heterogeneity of individual molecules throughout ongoing biological processes. A subset of single-molecule methods, single-molecule Forster Energy Resonance Transfer (henceforth smFRET), is used to measure nanometer distance changes between suitable fluorophore pairs, and has been invaluable in providing insights into structural biology and dynamic process that could not have been obtained via any other method. However, smFRET software remains fragmented; many existing tools must be used in combination, or individuals must develop bespoke software to be used within their group, meaning there are difficulties surrounding transfer and code maintenance if that individual leaves the research facility. 

The objective of this project was the development of smPy, an open-source Python package that aims to unify the smFRET data pre-processing, analysis, and visualisation pipeline. smPy's main strengths are, (1) its user-friendly interface in the manual peak selection and time-series display stages, (2) its flexible and adjustable background treatment, and (3) scope for three-colour FRET analysis. smPy is available under the GNU General Public License, Version 3 (GPLv3).

For smPy to benefit the single-molecule imaging community, functions must first be validated against standard molecule samples.
Additional next steps to improve smPy are, (1) the consideration of time efficiency, (2) full three-channel analysis capabilities, (3) batch analysis capability, and (4) a graphical user interface, GUI. 

The User Guides provide details on how to use smPy functions to analyse two colour and three-colour FRET data from start to finish, stages include:
* Importing and reading PMA files
* Creating mp4 files
* Creating an average frame
* Detecting `good' peaks
* Mapping channel peaks
* Background treatment
* Displaying and exporting intensity, FRET efficiency, and distance time-series plots.


## Dependencies

To install all required packages via pip:

```bash
pip install numpy scikit-image matplotlib opencv-python pillow


