Satellite Imagery Building Footprint Extraction

An end-to-end deep learning pipeline for extracting building footprints from high-resolution satellite imagery, utilizing a custom U-Net architecture.

Note: This codebase was originally developed as a proof-of-concept for the Gaia Lab at the University of Patras (August 2024).

Project Background & Vision

This project was conceptualized to assist in automated disaster response—specifically, assessing building damage following severe wildfires in Greece. The overarching vision involved a two-phase deep learning pipeline:

Extracting building footprints from pre-disaster Very High Resolution (VHR) satellite imagery.

Analyzing post-disaster imagery to automatically classify and map destroyed or damaged structures.

Due to data availability constraints for post-disaster VHR imagery, the project focused on mastering the critical first step. This repository represents the successful implementation of Phase 1: a robust pipeline for automated building footprint extraction, which serves as the foundational architecture for any subsequent damage classification tasks.

Project Overview

Identifying building footprints in aerial imagery is a highly imbalanced image segmentation problem, as buildings typically account for a small fraction of total pixels. This project implements a complete PyTorch pipeline to stream, process, and segment geospatial data, using a custom DiceBCE loss function to overcome the class imbalance.

Key Features

Optimized Data Pipeline: Implemented a CachedBuildingDataset class that streams the MapAI dataset directly from Hugging Face and caches training subsets into RAM, bypassing local hardware storage limits and network latency.

Custom U-Net Architecture: Built a lightweight convolutional neural network from scratch in PyTorch, tailored for binary semantic segmentation.

Imbalance Handling: Engineered a combined Dice + Binary Cross-Entropy (BCE) loss function to heavily penalize false negatives and force the model to recognize structural overlaps.

Adaptive Thresholding: Applied a statistical threshold (Mean + 1.5 Std Dev) to raw probability heatmaps to cleanly extract discrete building masks despite CPU-constrained training epochs.

Tech Stack

Deep Learning: PyTorch, Torchvision

Data Processing: Hugging Face Datasets, OpenCV, Rasterio, Shapely

Visualization: Matplotlib