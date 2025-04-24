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
* Detecting 'good' peaks
* Mapping channel peaks
* Background treatment
* Displaying and exporting intensity, FRET efficiency, and distance time-series plots.


## Dependencies

To install all required packages via pip:

```bash
pip install numpy scikit-image matplotlib opencv-python pillow
```

## Repository Structure

Below I provide a brief overview of the key directories within this repository:
### `pma files/`
- Contains the PMA files used within the user guides.

### `User Guide - Two Colour/`
- Contains a Jupyter Notebook and two interactive Python scripts.
- Walks through usage of the package for dual-channel (two-colour) FRET analysis.
- Ideal for hands-on learning or demonstration purposes.
- [`UserGuide_TwoColour.ipynb`](User%20Guide%20-%20Two%20Colour/UserGuide_TwoColour.ipynb): A Jupyter Notebook outlining the workflow for handling two-colour pre-processing, analysis, and visualisation.
- [`Manual_Select_TwoColour.py`](User%20Guide%20-%20Two%20/Manual_Select_TwoColour.py): An interactive script to aid in manually selecting donor-acceptor peak pairs.
- [`Time_Series_TwoColour.py`](User%20Guide%20-%20Two%20Colour/Time_Series_TwoColour.py): An interactive script to display time-series plots, upon selecting a mapped donor-acceptor pair.
- 
### `User Guide - Three Colour/`
- Contains a Jupyter Notebook and two interactive Python scripts.
- Walks through usage of the package for three-colour FRET analysis.
- Ideal for hands-on learning or demonstration purposes.
- [`UserGuide_ThreeColour.ipynb`](User%20Guide%20-%20Three%20Colour/UserGuide_ThreeColour.ipynb): A Jupyter Notebook outlining the workflow for handling three-colour pre-processing, analysis, and visualisation.
- [`Manual_Select_ThreeColour.py`](User%20Guide%20-%20Three%20/Manual_Select_ThreeColour.py): An interactive script to aid in manually selecting peak triplets.
- [`Time_Series_ThreeColour.py`](User%20Guide%20-%20Three%20Colour/Time_Series_ThreeColour.py): An interactive script to display time-series plots, upon selecting a mapped triplet.

### `Appendices/`
- Supplementary material, providing further explanations and figures.
- [`Appendix_A_Find_Peaks_Default_Values.ipynb`](Appendices/Appendix_A_Find_Peaks_Default_Values.ipynb): A Jupyter Notebook describing how the find_peaks() default argument values were arrived at.
- [`Appendix_B_LinearShift_Benefits.ipynb`](Appendices/Appendix_B_LinearShift_Benefits.ipynb): An Jupyter Notebook demonstrating the benefits of applying a linear shift prior to a polynomial mapping. 
- [`Appendix_C_3CHSyntheticFunction.ipynb`](Appendix_C_3CHSyntheticFunction.ipynb): An Jupyter Notebook outlining the function used to create synthetic 3CH PMA files.

