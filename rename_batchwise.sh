# this will remove "000" from all filenames. Exampele. filename000.png --> filename.png
for file in *; do mv "${file}" "${file/000/}"; done
