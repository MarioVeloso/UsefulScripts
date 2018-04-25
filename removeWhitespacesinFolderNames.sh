#!/bin/bash
#removeWhitespacesinFolderNames

# $1 is a directory containing subdirectories
# We would like to replace whitespaces in subdirectory names with underscores
# The variable "a" contains the directory name (i.e. $1)
# The variable "b" contains the subdirectory names
find $1 -depth -name "* *" -print0 | \
while read -d $'\0' f ; do
    a="$(dirname "$f")"
    b="$(basename "$f")"
    #optional check if the basename changes -> reduces errors in mv command
    #only needed when using -wholename instead of -name in find, so skippable
    if [ "${b// /_}" != "$b" ] ; then
    mv -v "$a"/"$b"  "$a"/"${b// /_}"
    fi
done