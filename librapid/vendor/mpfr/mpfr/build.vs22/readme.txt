
Licensing
---------

Where files in this distribution have been derived from files licensed
under Gnu GPL or LGPL license terms, their headers have been preserved 
in order to ensure that these terms will continue to be honoured.  

Other files in this distribution that have been created by me for use
in building MPIR and MPFR using Microsoft Visual Studio 2017 are 
provided under the terms of the LGPL version 2.1

Compiling MPFR with the Visual Studio 2017
------------------------------------------

The project files provided in this repository are intended for use with
any version of Visual Studio 2017;  they are designed for use with the
the development version of MPFR located at:

    https://gforge.inria.fr/scm/viewvc.php/mpfr/trunk/

and my deveopment version of MPIR at:

    https://github.com/BrianGladman/mpir

It is assumed that MPIR has already been built and that the directories
containing MPIR and MPFR are at the same level in the directory 
structure:

    mpir
        build.vc15
            dll     MPIR Dynamic Link Libraries 
            lib     MPIR Static Libraries
            ....
    mpfr
        build.vc15
            dll     MPFR Dynamic Link Libraries
            lib     MPFR Static Libraries
            ....

The root directory name of the MPIR version that is to be used in 
building MPFR should be 'mpir' with any version number such as in
'mpir-3.0' removed.
 
The full MPFR source distribution together with the Visual Studio
2017 build files can be obtained by cloning the GIT repository at:

    https://github.com/BrianGladman/mpfr
 
Alternatively the MPFR source distribution can be obtained from the
main MPFR development repository at:

    https://gforge.inria.fr/scm/viewvc.php/mpfr/trunk/


(see http://www.mpfr.org/ for more information).

After the MPFR source code has been placed into the MPFR root directory,
the build files should then be added by copying the sub-directory 
build.vc15 into the root directory as shown earlier. 

The root directory names 'mpir' and 'mpfr' are used because this makes 
it easier to use the latest version of MPIR and MPFR without having to 
update MPIR and MPFR library names and locations when new versions are 
released.
        
There are two build solutions, one for static libraries and the other 
for dynamic link libraries:

    lib_mpfr.sln    for static libraries
    dll_mpfr.sln    for dynamic link libraries

After loading the appropriate solution file the Visual Studio IDE allows
the project configuration to be chosen:

    win32 or x64
    release or debug
    
after which the lib_mpfr library should be built first (but see Tuning
below), followed by lib_tests (under lib_mpfr_tests or dll_mpfr_tests) 
and then the tests themselves.

If you wish to use the Intel compiler, you need to convert the build
files by right clicking on the MPFR top level Solution and then selecting
the conversion option.

Any of the following projects and configurations can now be built:

    dll_mpfr    the DLL (uses the MPIR DLL) 
      Win32
        Debug
        Release
      x64
        Debug
        Release

    lib_mpfr    the static library (uses the MPIR static library) 
      Win32
        Debug
        Release
      x64
        Debug
        Release

The library outputs are placed in the directories:

    mpfr
        lib
        dll 

Tuning
------

Because tuning is not reliable on Windows, tuning parameters are picked
up from the *nix builds. 

Before building MPFR, the choice of architecture for tuning should be
selected by editing the mparam.h file in the build.vc15 directory to
select the most appropriate tuning parameters.

Test Automation
----------------

Once the tests have been built the scripts run_lib_tests.py or
run_dll_tests.py (in the build.vc15 folder) can be used to run them. 
If Python is not installed the tests have to be run manually.

Acknowledgements
----------------

My thanks to:

1. The GMP team for their work on GMP and the MPFR team for their work 
   on MPFR
2. Patrick Pelissier, Vincent Lefèvre and Paul Zimmermann for helping
   to resolve VC++ issues in MPFR.
3. The MPIR team for their work on the MPIR fork of GMP.
4. Jeff Gilcrist for his help in testing, debugging and improving the
   readme.txt file giving the build instructions
5. Alexander Shevchenko for helping in tidying up the build projects.
 
     Brian Gladman, August 2017 

