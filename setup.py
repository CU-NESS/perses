#!/usr/bin/env python

import os, urllib

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

packages = ['perses.util', 'perses.models', 'perses.beam',\
    'perses.foregrounds', 'perses.simulations']
        
setup(name='perses',
      version='0.1',
      description='Pre-EoR Signal Extraction Software',
      packages=packages,
     )

PERSES_env = os.getenv('PERSES')
cwd = os.getcwd()

##
# TELL PEOPLE TO SET ENVIRONMENT VARIABLE
##
if not PERSES_env:

    import re    
    shell = os.getenv('SHELL')

    print("\n")
    print("#" * 78)
    print("It would be in your best interest to set an environment variable")
    print("pointing to this directory.\n")

    if shell:    
        if re.search('bash', shell):
            print("Looks like you're using bash, so add the following to " +\
                "your .bashrc:")
            print("\n    export PERSES={0}".format(cwd))
        elif re.search('csh', shell):
            print("Looks like you're using csh, so add the following to " +\
                "your .cshrc:")
            print("\n    setenv PERSES {!s}".format(cwd))        

    print("\nGood luck!")
    print("#" * 78)
    print("\n")

# Print a warning if there's already an environment variable but it's pointing
# somewhere other than the current directory
elif PERSES_env != cwd:

    print("\n")
    print("#" * 78)
    print("It looks like you've already got an PERSES environment variable " +\
        "set but it's \npointing to a different directory:")
    print("\n    PERSES={!s}".format(PERSES_env))

    print("\nHowever, we're currently in {!s}.\n".format(cwd))
    
    print("Is this a different ares install (might not cause problems), or " +\
        "perhaps just")
    print("a typo in your environment variable?")

    print("#" * 78)
    print("\n")
    
    
    
