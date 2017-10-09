# Machine learning pipeline for CellProfiler output

This is a vignette of using Scikit-learn machine learning library to perform supvervised and unsupervised classification of cellular phenotypes, as a downstream step from CellProfiler analysis (output as SQLite database)

# Installation

You will need Python and its machine learning libraries

1. Install Python (>=2.7.9 or >=3.4) from https://www.python.org/downloads/

    In case you already had older version of Python that does not include pip, and you do not wish to upgrade, please follow the instruction at: https://packaging.python.org/tutorials/installing-packages/
 
1. Open a command line window
    
    In Linux/Mac OS, open the "Terminal".
    In Windows, open the "cmd" (as administrator) here's how-to: https://www.howtogeek.com/235101/10-ways-to-open-the-command-prompt-in-windows-10/

1. In Terminal/cmd, type :
    ``` r
    pip3 install --upgrade pip
    ```
    In OSX/Linux you may have to use "sudo", e.g: 
    ``` r
    sudo pip3 install --upgrade pip
    ```
    If you prefer to use Python 2, then run with "pip" instead of "pip3", e.g.: 
    ``` r
    sudo pip install --upgrade pip
    ```

1. If success, continue to run the following commands, one line at a time:
    ``` r
    pip3 install numpy
    pip3 install scipy
    pip3 install sklearn
    pip3 install pandas
    pip3 install matplotlib
    pip3 install sqlite3
    pip3 install jupyter
    ```

1. Once done, test if you can use jupyter notebook. In Terminal or cmd, type:
    ``` r
    jupyter notebook
    ```
    If it opens an interface in your default web-browser, you’re ready!

    If you have issue running jupyter, please visit: https://jupyter.readthedocs.io/en/latest/install.html

    The script can also be executed by command line (if jupyter is not available).

    You can test run if you wish. Beware ! There will be (slightly) heavy computational tasks, please free your computer CPU and memory before running the script.

    To test run with command line (in Terminal or cmd), navigate to the folder location of the script.
    ``` r
    cd /path_to_YOUR_folder/MachineLearning
    python MLCP.py
    ```

The script in its current stage will need to be placed in a folder in the same level with the CellProfiler output, i.e. “CPOut” and “MachineLearning” folders.

The script will also look for files named “DefaultDB_train.db” and “DefaultDB_test.db” inside “CPOut”. Please consider this if you need to change the CellProfiler Output.

If you wish, you can change these path and names easily by editing the script itself using any text editor.
