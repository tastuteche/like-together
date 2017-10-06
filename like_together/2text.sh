#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

origin_file=$(realpath "$1")
origin_dir=$(dirname "$origin_file")
origin_basename=$(basename "$origin_file")
mkdir -p "$origin_dir/_textract_/"
text_file="$origin_dir/_textract_/${origin_basename}.txt"
textract "$origin_file" > "$text_file"
