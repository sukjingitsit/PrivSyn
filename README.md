# PrivSyn: Differentially Private Data Synthesis
This code has been published as an implementation of PrivSyn: Differentially Private Data Synthesis for a course project.

To run the code, one has to perform two major steps,
* Dataset Preparation
* Notebook Preparation

## Dataset Preparation
This task is not covered within the code and has to be performed manually specific to each dataset beforehand. There are three datasets used within the code 'Polish', 'UCI Adult' and 'US Accidental Drug Deaths'. These may be referenced for formatting. Essentially, each dataset needs the corresponding csv file, and three configuration related files. These configuration files are necessary for data loading and postprocessing after the algorithm is run. Each of the configuration files is used in the code as follows,
* loading_data.json is used for an overview of each attribute
* data.yaml is used to handle identifier attributes, numeric attributes (binning) and also grouping attributes (which is implemented but commented out as it seldom leads to errors)
* column_info.json is used for datatypes and nan-value handling

## Notebook Preparation
Again, the sample notebooks for three datasets have been attached. In those files, primarily the Dataloader class call and Anonymiser class call need to be modified in accordance with the appropriate file paths and epsilon/delta for differential privacy.

## Code Structure
The flow of code is as follows,
* Dataset-related files are passed to dataloading which first converts them to an appropriate form, such as dealing with missing, categorial and continuous values/attributes, and then calculates the indifference metric for all pairs of attributes (and all inherently included pre-processing such as one-way and two-way marginals).
* The next function (Anonymiser) applies noising to the dataset, which includes one-way marginal noising by the Gaussian Mechanism, two-way marginal selection (DenseMarg), marginal combination and two/multi-way marginal noising by the Gaussian Mechanism
* The next call is to Consistenter which is used to consist the marginals so as to satisfy the requirements of a probability distribution using Norm-Sub for l_2 consistency
* Thereafter, a call is made to the Gradually Update Method (GUM) which also has a subfolder. The subfolder consists of View.py (which is a class defined for ease of processing), ViewConsistenter.py (which is defined for a second iteration of marginal consistenting before GUM is called) and RecordSynthesiser.py which is the main call to the GUM method
* Finally, PostProcessor.py converts the pandas dataset back to the same format in which the data was input (such as unbinning continuous attributes or decoding categorical attributes)
Other files by the name of tempAnonymisation.py and main.py are present as initial versions of the code which were then removed in favour of Anonymisation.py and using separate .ipynb notebooks for each dataset (as opposed to a command-line interface).

## Requirements
The following libraries are required for the functioning of the code,
* Pyyaml
* Numpy
* Pandas
* Networkx
* Scikit-Learn

The code was run a MacOS with osx-arm64 compiler

## Acknowledgements
A code implementation of DPSyn, an algorithm similar to PrivSyn and developed by the same authors, is also available on Github at [this link](https://github.com/agl-c/deid2_dpsyn)
