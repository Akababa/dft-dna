# dft-dna

## Steps
1. Use `clean.cpp` to convert the original fasta files into a format with one sequence per line. (`cleaned/` folder)
2. Use `main.py` to do the rest. I originally tried to do it all in fft.cpp but it's much easier to use sklearn.

## `ACGTClassifier` class
* Implements the sklearn estimator interface, and properly handles separation of training/test when given raw sequences.
* Allows ACGT values to be set as estimator params.

## `main.py`
* Simple 10-fold cross-validation for list of sklearn estimators and one value.
* Grid search to find optimal ACGT values, and outputs results to csv file.
