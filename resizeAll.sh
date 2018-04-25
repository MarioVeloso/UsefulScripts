  #!/bin/bash
#Resize all images in a Folder using imagemagick

#Get a file as an input - if it is not a file it will not work
[ $# -ge 1 -a -f "$1" ] && input="$1" || input="-"
cat $input

# The exclamation point after the size force the size without respectinf aspect ratio - remove it if you want to respect aspect ratio
for i in input/*.JPG;
do
  convert $i -resize 28x28! $(basename $i .JPG).JPG;
done
