#!/bin/bash

JOB_WORKING_DIR=/home/andres_torres_uri_edu/ondemand/data/sys/myjobs/projects/default/3

cd $JOB_WORKING_DIR

shopt -s nullglob
files=( *.out )
shopt -u nullglob

for f in "${files[@]}"; do
    echo "Removing Output File: $f"
    rm "$f"
done
