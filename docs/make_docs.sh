#!/bin/bash

if [ -n "$DISTPY" ] && [ -n "$PYLINEX" ] && [ -n "$PERSES" ]
then
    cd $PERSES/docs
    pdoc --config latex_math=True --html $DISTPY/distpy $PYLINEX/pylinex $PERSES/perses --force
    cd - > /dev/null
else
    echo "DISTPY, PYLINEX, and PERSES environment variables must be set for the make_docs.sh script to be used to make the documentation."
fi
