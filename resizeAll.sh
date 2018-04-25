#!/bin/bash
#Resize all images in a Folder

#Get a file as an input - if it is not a file it will not work
[ $# -ge 1 -a -f "$1" ] && input="$1" || input="-"
cat $input

for i in input/*.JPG;
do
convert $i -resize 28x28 $(basename $i .JPG).JPG;
done
