#!/bin/bash

#using imagemagick
mogrify -format jpg ./trainA/*.png
rm ./trainA/*.png
mogrify -format jpg ./trainB/*.png
rm ./trainB/*.png
