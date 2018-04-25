#!/bin/bash 

for file in $1/*.png;
do
    cp "$file" $2;
done
