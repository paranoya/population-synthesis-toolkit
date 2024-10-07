#!/bin/sh

echo "Input data directory: $1"

for file in `ls $1/*.ised`
do
    export name=$file
    echo "Running GALAXEVPL for file: $name"
    $bc03/galaxevpl $name -all

    echo "Saving output in fits format"
    python $bc03/to_fits.py $name
done

