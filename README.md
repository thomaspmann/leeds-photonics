# Leeds-photonics

===========
This is a Python 3+ library for the Applied Photonics group at the University of Leeds.

Contains:
* Fluorescence fitting algorithms for spectra and time resolved decay measurements
* CAD files for in house equipment accessories (located in 'other' folder)

See "Examples" folder for scripts that use the functions in this module.


Installation Information from PyPI
----------------------------------

You can install the program by typing the following into cmd.exe (windows) or terminal (mac):
```
pip install photonics
```


Installation Information from GIT
---------------------------------
If you will be getting updated code from git, use git clone to put the directory
somewhere. Then do the following to generate a link to your git directory:
```
python setup.py develop
```

If you want the normal installation (e.g. copies files to Python installation) use:
```
python setup.py install
```


A Note on Python Installation
---------------------------------
Modern computers have python built into them so the above commands should usually work without any other requirements. 

I recommend installing Anaconda (https://www.continuum.io/downloads, Python 3.6+ version) as a package 
manager to help keep things up to date. It also comes with a set of commonly used scientific packages so that they
do not need to be installed individually in the future. A quickstart guide on how to use anaconda can be found at
https://conda.io/docs/using/cheatsheet.html .

For workflow, i'd advise using jupyter notebook for small exploratory coding and the IDE PyCharm for heavier stuff.
The former can be opened by typing 

```
jupyter notebook
```

into cmd.exe or terminal. PyCharm community version can be installed at 
https://www.jetbrains.com/pycharm/download/#section=windows for free.


Authors, Copyright, and Thanks
------------------------------
lifetime is Copyright (C) 2017 By:
 * Thomas Mann <mn14tm@leeds.ac.uk>
 
 All rights reserved.
See LICENSE.md for license terms.

Contributing
------------------------------
1. Fork.
2. Make a new branch.
3. Commit to your new branch.
4. Add yourself to the authors/acknowledgements (whichever you find appropriate).
5. Submit a pull request.
