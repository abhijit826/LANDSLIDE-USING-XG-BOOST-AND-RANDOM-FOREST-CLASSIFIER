# Landslide Susceptibility Mapping

This project demonstrates a workflow for creating landslide susceptibility maps using machine learning, specifically the XGBoost algorithm, and geospatial data (Digital Elevation Models). The project focuses on the Kinnaur region and utilizes a combination of real landslide inventory data and synthetic data for model training and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Data](#data)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Susceptibility Mapping](#susceptibility-mapping)
- [Visualization and Analysis](#visualization-and-analysis)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

The goal of this project is to develop a predictive model that can identify areas prone to landslides based on various environmental factors. The key steps involve:
1.  **Data Collection and Preparation:** Gathering landslide inventory data and relevant environmental factors (DEM, rainfall, soil type, etc.).
2.  **Feature Extraction:** Extracting relevant features from geospatial data, such as elevation and slope from DEMs.
3.  **Dataset Creation:** Combining landslide data with features to create a labeled dataset for model training, including generating synthetic data to augment limited real data.
4.  **Model Selection and Training:** Choosing and training a suitable machine learning model (XGBoost) for landslide prediction.
5.  **Hyperparameter Tuning and Evaluation:** Optimizing model performance using techniques like GridSearchCV and K-Fold Cross-Validation.
6.  **Susceptibility Mapping:** Applying the trained model to DEM data to generate landslide susceptibility maps.
7.  **Visualization and Analysis:** Visualizing results, including feature importance and spatial susceptibility patterns.

## Methodology

The project employs the following key techniques:

-   **Geospatial Data Processing:** Using `rasterio` to handle DEM files and extract topographical features (elevation, slope).
-   **Data Wrangling:** Utilizing `pandas` for data loading, cleaning, and manipulation.
-   **Synthetic Data Generation:** Creating synthetic data to address the limited size of the real landslide inventory and improve model robustness.
-   **Machine Learning (XGBoost):** Implementing an XGBoost classifier for binary prediction (landslide/no landslide).
-   **Model Evaluation:** Employing `scikit-learn` for splitting data, cross-validation (K-Fold), hyperparameter tuning (GridSearchCV), and performance reporting (classification report, confusion matrix, F1 score).
-   **Visualization:** Generating various plots (`matplotlib`, `seaborn`) including feature importance, correlation matrices, scatter plots, and susceptibility maps.

## Data

The project uses:

-   **Consolidated Landslide Inventory:** A pandas DataFrame containing real landslide locations and attributes (Date, Latitude, Longitude, District, Trigger).
-   **Digital Elevation Models (DEMs):** GeoTIFF files (`.tif`) representing the topography of the study area (Kinnaur region).
-   **Synthetic Dataset:** A generated dataset to supplement the real data and improve model training.
-   **Rainfall Data:** Time series data (`.csv`) for rainfall information.

*(Note: The actual data files like DEMs and `PERSIANN_20200311_20250730.csv` are assumed to be uploaded to the Colab environment or specified paths.)*

## Model Training and Evaluation

The project uses an XGBoost Classifier for landslide prediction.

-   **Training Data:** A combined dataset of real landslide locations and synthetically generated non-landslide locations, along with extracted features (Elevation, Slope_Angle, Rainfall_mm, Soil_Saturation, Vegetation_Cover, Earthquake_Activity, Proximity_to_Water, Soil_Type_Gravel, Soil_Type_Sand, Soil_Type_Silt).
-   **Hyperparameter Tuning:** GridSearchCV is used to find the optimal hyperparameters for the XGBoost model, aiming to maximize the F1 score, which is suitable for imbalanced datasets.
-   **Cross-Validation:** K-Fold Cross-Validation is performed on the best model to provide a more robust estimate of its performance.

## Susceptibility Mapping

After training, the best model is applied to the DEM data to predict landslide susceptibility scores for each pixel.

-   For each DEM file, Elevation and Slope are extracted.
-   Mean values from the training data are used for non-topographical features (Rainfall, Soil Saturation, etc.) as a simplified approach for pixel-wise prediction across the entire DEM.
-   The model predicts a probability (susceptibility score) for each pixel indicating the likelihood of a landslide.
-   Susceptibility maps are visualized using a color gradient from low (blue) to high (red) susceptibility.

## Visualization and Analysis

The project includes several visualizations to understand the data and the model's behavior:

-   **Feature Importance:** A bar chart showing the relative importance of each feature in the XGBoost model's predictions.
-   **Correlation Heatmap:** A heatmap illustrating the correlations between different features and the landslide target variable.
-   **Scatter Plots, Box Plots, and Violin Plots:** Visualizations to explore the relationships and distributions of key features with respect to landslide occurrence.
-   **Susceptibility Maps:** Visual representations of the predicted landslide susceptibility across the DEMs.
  ## LANDSLIDE PREDICTION USING PICKLE FILE IN LOCAL HOST
<img width="1750" height="946" alt="Screenshot 2025-11-07 221250" src="https://github.com/user-attachments/assets/0dd4bab1-8fbc-40e3-be32-68d19d99c862" />
<img width="1728" height="968" alt="Screenshot 2025-11-07 221415" src="https://github.com/user-attachments/assets/2d32a884-4f4b-4cc7-9282-d51bfc52085f" />

## Getting Started

These instructions will get you a copy of the project up and running on your local machine or in a Colab environment for development and testing purposes.

### Prerequisites

-   Python 3.7+
-   Google Colab environment (recommended for ease of use with Colab notebooks and file uploads) or a local Python environment.
-   Required Python libraries (listed in `requirements.txt`).
-   DEM files (`.tif`) for the area you want to analyze.
-   Rainfall data file (`.csv`) - the notebook is configured to use `PERSIANN_20200311_20250730.csv`.
-   Optional: A generic landslide dataset file (`landslide_dataset.csv`) if you want to run the combined data cells.
