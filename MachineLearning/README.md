These are the scripts to perform feature selection/preprocessing (step 2) and machine learning classification (step 3) on the .DB output of CellProfiler
  **Input of this step:** 

     * SQLite database .DB files (from step 1)
     
  **Output of this step:** 

     * Histogram of feature ranking (.PNG)
     * Data table of objects with selected features (.TXT files)
     * Metadata labels of objects (.TSV file)
     
     * Plot of the recalls (.PNG), i.e. percent of cells correctly classified
     * Confusion matrix (.PNG and .TXT) 
     * Weights of the trained models (.SAV)

  **Note** In this task, there might be heavy computational tasks, depends on how large your data is. Please save your current works, free your computer CPU and memory before running the script.

  To run the script (for step 2 and 3), please use:

  Jupyter notebook for **MachineLeaning**/*MLCP.ipynb* 

  ``` r
  jupyter notebook
  ```

  or command line for **MachineLeaning**/*MLCP.py* (if jupyter is not available):

  ``` r
  cd /path_to_YOUR_folder/MachineLearning
  python MLCP.py
  ```   

Another vignettes of similarity can be viewed at:

http://cellprofiler.org/imagingflowcytometry/

https://github.com/holgerhennig/machine-learning-IFC

https://github.com/cytomining/cytominer/blob/master/vignettes/cytominer-pipeline.Rmd
