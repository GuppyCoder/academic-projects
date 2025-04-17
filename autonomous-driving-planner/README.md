## Academic Integrity Notice

This repository is made public solely for portfolio and educational demonstration purposes.  
All code in this repository was completed as part of a graduate-level course at UT Austin.

Any reuse, copying, or submission of this code for academic credit is a violation of university academic integrity policies.

If you are a student currently taking a similar course, **do not copy or submit any part of this code.**

## Attribution

Some components of this project (such as evaluation scripts and dataset loaders) were provided as part of the official course materials at UT Austin.  
They are included here solely for demonstration purposes to provide context for the models and code I developed during the course.

Again, If you are a current student in this or a similar course, **do not reuse or submit any of this code.**



# Autonomous Driving Planner

This project implements multiple models to predict vehicle trajectories based on lane boundary inputs and raw images, using supervised deep learning techniques. The models were developed as part of graduate coursework in deep learning and computer vision.

## Overview

The goal of this project is to train planners that predict future waypoints for a vehicle to follow, enabling autonomous driving in a simulated environment.

The following models were implemented:
- **MLP Planner:** A Multi-Layer Perceptron model that predicts waypoints from ground truth lane boundary points.
- **Transformer Planner:** A Transformer-based model that uses learned waypoint embeddings to attend over lane boundaries and predict future vehicle positions.
- **CNN Planner:** A Convolutional Neural Network model that predicts waypoints directly from RGB images captured from the vehicle's perspective.

## Project Structure


Key components:
- `models.py`: Contains the implementations of the MLP, Transformer, and CNN planners.
- `metrics.py`: Evaluation metrics for assessing model performance.
- `datasets/`: Data loading and transformation modules.
- `supertux_utils/`: Tools for visualizing and evaluating the driving behavior.

## Dataset

The models are trained and evaluated on the [SuperTuxKart Drive Dataset](https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip), a driving simulation dataset with labeled trajectory data.

## Training and Evaluation

Each model was trained from scratch using a custom training pipeline. Performance was measured using:
- **Longitudinal Error:** Measures accuracy along the vehicle's direction of travel.
- **Lateral Error:** Measures accuracy perpendicular to the direction of travel (i.e., left/right deviations).

Models achieving low longitudinal and lateral errors were able to successfully drive the vehicle autonomously in the SuperTuxKart simulation environment.

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- PySuperTuxKart (optional for simulation visualization)


## Notes

This project was completed as part of the M.S. Computer Science curriculum at the University of Texas at Austin. Some dependencies and simulation visualizations require additional setup and are optional for basic model training and evaluation.
