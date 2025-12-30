---
title: SAM3 Image Segmentation
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: gradio_app.py
pinned: false
license: mit
---

# SAM3 Image Segmentation Studio

A simple and elegant image segmentation tool powered by **SAM3 (Segment Anything Model 3)**.

## Features

- **Text-based Segmentation**: Describe objects in natural language
- **Transparent Background Export**: All extracted objects are saved as PNG with transparent background
- **Batch Download**: Download all segmented objects as a ZIP file

## Usage

1. Upload an image
2. Enter what you want to segment (e.g., "cat", "person", "car")
3. Click "Start Segmentation"
4. Download the extracted objects

## Model

This Space uses the official SAM3 model from [facebook/sam3](https://huggingface.co/facebook/sam3).
