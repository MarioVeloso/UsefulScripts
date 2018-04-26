#!/bin/bash

# Usage
cat "usage : sh pngtojpg.sh path/to/folder/"

# Using imagemagick
mogrify -format jpg $1*.png
rm $1*.png
