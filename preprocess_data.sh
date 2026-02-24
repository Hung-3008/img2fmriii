#!/bin/bash

# Preprocess data for subjects 2, 5, and 7
SUBJECTS=(2 5 7)

echo "Starting data preprocessing pipeline..."

for sub in "${SUBJECTS[@]}"; do
    echo "================================================================"
    echo "Processing Subject ${sub}"
    echo "================================================================"
    
    # 1. Prepare NSD data (fMRI and Stimuli), using zscore matching train config
    echo "-> Running prepare_nsddata_scale.py for subj0${sub} (z-score mode)"
    python src/data/prepare_nsddata_scale.py -sub ${sub} -session 40 -mode zscore
    if [ $? -ne 0 ]; then
        echo "Error in prepare_nsddata_scale.py for sub ${sub}!"
        exit 1
    fi

    # 2. Extract PNG images from validation arrays and build eval tensor
    echo "-> Running save_images.py for subj0${sub}"
    python src/data/save_images.py --sub ${sub}
    if [ $? -ne 0 ]; then
        echo "Error in save_images.py for sub ${sub}!"
        exit 1
    fi
done

echo "================================================================"
echo "Extracting Multi-Layer DINOv2 Features for Subjects: ${SUBJECTS[*]}"
echo "================================================================"

# 3. Extract DINOv2 ViT-B/14 features using the multi-layer script
python src/data/extract_features_dinov2_multilayer.py --subjects ${SUBJECTS[@]}
if [ $? -ne 0 ]; then
    echo "Error in extract_features_dinov2_multilayer.py!"
    exit 1
fi

echo "All preprocessing steps completed successfully!"
