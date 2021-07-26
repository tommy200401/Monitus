#!/bin/bash
for FILE in *.csv
do
    if [ ! -f "$FILE.zst" ]; then
        echo Processing $FILE
        zstd "$FILE" "$FILE.zst"
    fi
done
