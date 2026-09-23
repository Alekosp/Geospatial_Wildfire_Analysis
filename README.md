Geospatial Wildfire Analysis & Infrastructure Damage Assessment

An end-to-end research initiative transitioning from optical remote sensing analysis in QGIS to automated deep learning computer vision pipelines in Python.

Project Background

This repository stems from my Master's thesis research (2023) and subsequent work as an External Research Collaborator at the University of Patras' Gaia Laboratory (2024).

Initially, the research focused on quantifying infrastructure loss from the devastating 2021 Greek wildfires (Ziria, Northern Evia, and Eastern Attica). Using Copernicus Sentinel-2A satellite imagery and QGIS, I calculated the Normalized Burn Ratio (dNBR) to delineate burn scars and assess damage to road networks and buildings.

Building on this academic foundation, my subsequent work at the Gaia Laboratory focused on leveraging Deep Learning for disaster response. Utilizing Very High Resolution (VHR) satellite imagery, I developed a Python-based framework to automate the extraction of baseline infrastructure—the critical first step in automatically detecting fire-damaged buildings.

Furthermore, driven by personal curiosity to fully automate spatial workflows, I am currently translating the original QGIS-based methodologies (such as Sentinel-2 burned area extraction and spatial intersections) into pure Python pipelines within this repository.

Deep Learning Building Extraction

The main notebook in this repository (Unet_building_footprint_extraction.ipynb) contains the PyTorch pipeline designed to extract building footprints from aerial imagery, addressing the inherent class imbalance of geospatial segmentation:

Optimized Data Pipeline: Implemented a CachedBuildingDataset class that streams the MapAI dataset from Hugging Face and caches training subsets into RAM.

Custom U-Net Architecture: Built a lightweight convolutional neural network from scratch for binary semantic segmentation.

Imbalance Handling: Engineered a combined Dice + Binary Cross-Entropy (BCE) loss function to heavily penalize false negatives.

Adaptive Thresholding: Applied statistical thresholding (Mean + 1.5 Std Dev) to cleanly extract discrete building masks from probability heatmaps.

Publications

The methodologies and findings associated with this research have been peer-reviewed and published:

A. Paganias et al., "Mapping recent wildfires in Greece and the associated built-up losses," Proc. SPIE, 2024. https://doi.org/10.1117/12.3031800

A. Paganias, C. Pappas, "Harvesting remote sensing observations for quantifying burned area and built-up losses from the 2021 wildfires in Greece," Proc. SPIE 12735, 2023. https://doi.org/10.1117/12.2680887

Tech Stack

GIS & Remote Sensing: QGIS, Copernicus Sentinel-2A, Spatial Intersection, dNBR Indexing

Deep Learning: PyTorch, Torchvision, Python

Spatial Data Processing: Hugging Face Datasets, OpenCV, Rasterio, Shapely