# Dexter Lab Facial Detection and Recognition

This repository contains a project that implements two computer vision tasks on images from the Dexter cartoon:

1. **Task 1**: Detect all faces in the images
2. **Task 2**: Detect faces of the four main characters: Dexter, Deedee, Mom, and Dad

The project includes two approaches:
- A solution from scratch (no pretrained models) using a sliding window technique
- A solution using a pretrained YOLO model

---

## Project Structure

- **`facial_detector.py`**: Implements ML models for the tasks
- **`parameters.py`**: Contains configuration parameters for the models
- **`process_images.py`**: Helper functions for processing images
- **`visualize.py`**: Helper functions for visualizing detections
- **`run_project.py`**: Main script to run the project and generate test results
- **`evaluare/`**: Folder containing the evaluation script
- **`models/`**: Folder to save the best models
- **`testare/`**: Folder containing test images

---

## Prerequisites

To run this project, you need the following:

- **Python**: Version 3.12.7
- **Required Libraries**: Install the dependencies in `requirements.txt`

---

## Documentation

More information about the project is in the documentation `documentation.pdf`