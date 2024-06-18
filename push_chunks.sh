#!/bin/bash

# Function to add, commit, and push if files exist
function add_commit_push {
    file_type=$1
    message=$2
    if ls $file_type 1> /dev/null 2>&1; then
        git add $file_type
        git commit -m "$message"
        git push origin main
    else
        echo "No $file_type files to add."
    fi
}

# Model files
add_commit_push "*.model" "Add model files"

# JSON files
add_commit_push "*.json" "Add JSON files"

# BIN files
add_commit_push "*.bin" "Add BIN files"

# SAFETENSORS files
add_commit_push "*.safetensors" "Add SAFETENSORS files"

# H5 files
add_commit_push "*.h5" "Add H5 files"

# PT files
add_commit_push "*.pt" "Add PT files"

# DAT files
add_commit_push "*.dat" "Add DAT files"

# CSV files
add_commit_push "*.csv" "Add CSV files"

# TSV files
add_commit_push "*.tsv" "Add TSV files"

# Image files
add_commit_push "*.jpg" "Add JPG files"
add_commit_push "*.png" "Add PNG files"
add_commit_push "*.gif" "Add GIF files"

# Archive files
add_commit_push "*.zip" "Add ZIP files"
add_commit_push "*.tar.gz" "Add TAR.GZ files"

# Executable files
add_commit_push "*.exe" "Add EXE files"
add_commit_push "*.dll" "Add DLL files"

# Finally, stage, commit, and push remaining files
git add .
git commit -m "Add remaining files"
git push origin main
