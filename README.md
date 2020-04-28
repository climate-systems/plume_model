# plume_model
A simple model to compute plume thermodynamic properties upon lifting and mixing

Ensure that python is installed on your machine with requisite libraries.

Steps to run plume model:
  i) Ensure that you have access to ARM_Nauru directory (for input files). 
       Else modify code to point to right input
  ii)  Compile the cython libraries with "python setup.py --build_ext -inplace" in your terminal
  iii) If cython build is successful, run "python plume_model.py" in your terminal. 

