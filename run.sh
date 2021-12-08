#!/bin/sh

../rai -p SLIC_CUDA 2>&1 | tee raioutput.txt

URL=$(echo $(grep -oP 'http.?://\S+' raioutput.txt) | rev | cut -c 2- | rev)

wget $URL

FILE=$(ls -d build*)

tar -xvf $FILE


 