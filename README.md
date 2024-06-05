# plume_model
A simple model to compute plume thermodynamic properties upon lifting and mixing.
Written by: Fiaz Ahmed (exhaustive rewrite of legacy codes in IDL).

Ensure that python is installed on your machine with requisite libraries (see requirements.txt).
You can directly install these modules using ***conda install --yes --file requirements.txt*** or ***pip install -r requirements.txt**


Steps to run plume model:

  i)  Compile the cython libraries with "python setup.py build_ext --inplace" in your terminal. This will compile the code in thermo_functions.pyx. 
       Cython significantly speeds up the numerical computation.

  ii) If cython build is successful, interact with the Jupyter notebook <mark>run_plume_model.ipynb</mark>. Ensure that you point the code to the correct
      input directories and the appropriate preprocessors. The default version of the notebook is designed to 
      run on ARM data, particularly using data at two sites: Nauru and CACTI.
