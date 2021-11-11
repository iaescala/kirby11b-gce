# kirby11b-gce
Python implementation of the Kirby+11b galactic chemical evolution model (including [X/Fe])

The original GCE code was written in IDL by E. Kirby (Caltech/Notre Dame). G. Duggan (Caltech/IPAC) rewrote the code in Python. 
I. Escala (Caltech/Carnegie) has modified G.Duggan's Python verison of the model to include additional options such as more IMFs and DTDs.
I have also included a module for fitting the GCE model using minimization algorithms (GCE_FIT_LIKE) based on E. Kirby's original IDL code.

# Usage #

See the example in the included Jupyter notebook.

# Requirements #

Python 3, standard libraries such as numpy, scipy

# Citation #

Please cite Kirby et al. 2011b and include a link to this repository 
