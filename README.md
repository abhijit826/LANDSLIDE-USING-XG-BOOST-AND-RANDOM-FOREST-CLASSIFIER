# 🌍 Landslide Susceptibility Score & Mapping Prediction  using Machine Learning 
NOTE-College/Office Wifi not Supported
🚀 **An intelligent geospatial workflow for predicting landslide-prone areas using XGBoost and QGIS-based feature engineering (LULC, DEM, slope, rainfall, etc.).**  
📍 **Study Area:** Kinnaur Region, Himachal Pradesh, India  

---

## 🧭 Table of Contents  
- [📘 Project Overview](#-project-overview)  
- [⚙️ Methodology](#️-methodology)  
- [🗂️ Data](#️-data)  
- [🤖 Model Training and Evaluation](#-model-training-and-evaluation)  
- [🗺️ Susceptibility Mapping](#️-susceptibility-mapping)  
- [📊 Visualization and Analysis](#-visualization-and-analysis)  
- [🚀 Getting Started](#-getting-started)  
- [🧰 Prerequisites](#-prerequisites)  
- [💻 Installation](#-installation)  
- [🪄 Usage](#-usage)  
- [📁 File Structure](#-file-structure)  
- [📜 License](#-license)  
- [🙏 Acknowledgments](#-acknowledgments)  

---

## 📘 Project Overview  

The goal of this project is to develop a **predictive geospatial model** that identifies areas prone to landslides based on multiple environmental and terrain-based parameters.  

### 🧩 Key Steps:
1. 🗺️ **Data Collection & Preparation:** Landslide inventory + DEM, rainfall, soil, and vegetation data.  
2. 🧮 **Feature Extraction:** Extracting elevation, slope, aspect, and NDVI from DEMs and remote sensing data.  
3. 🧠 **Model Training:** Using **XGBoost Classifier** to predict landslide likelihood.  
4. 🎯 **Evaluation:** Model performance optimized with **GridSearchCV** & **K-Fold Cross-Validation**.  
5. 🌈 **Mapping:** Generating landslide susceptibility maps using pixel-wise model inference on DEMs.  

---

## ⚙️ Methodology  

The workflow integrates **geospatial analysis** and **machine learning** as follows:  

- 🗻 **Geospatial Processing:** Using `rasterio` and `GDAL` for DEM handling and terrain metrics.  
- 🧹 **Data Wrangling:** `pandas` and `numpy` for preprocessing, merging, and transformation.  
- 🤖 **Machine Learning (XGBoost):** Binary classification for landslide/no-landslide detection.  
- 🔍 **Model Evaluation:** Confusion matrix, precision, recall, F1-score, ROC curve.  
- 🎨 **Visualization:** `matplotlib` and `seaborn` for plots and correlation heatmaps.  

---

## 🗂️ Data  

### 📁 Datasets Used:
- 🪨 **Landslide Inventory:** Real-world data (Date, Lat, Lon, District, Trigger).  
- 🗺️ **DEM Files:** Topography raster data for the Kinnaur region.  
- 🧪 **Synthetic Data:** Generated to balance landslide/non-landslide samples.  

---

## 🤖 Model Training and Evaluation  

- 🧠 **Model:** `XGBoostClassifier`  
- 📈 **Training Data:** Combination of real and synthetic landslide samples.  
- ⚙️ **Features:**  
  - Elevation  
  - Slope Angle  
  - Rainfall (mm)  
  - Soil Saturation  
  - Vegetation Cover (NDVI)  
  - Earthquake Activity  
  - Proximity to Water  
  - Soil Type (Gravel, Sand, Silt)  

- 🧩 **Tuning:**  
  - `GridSearchCV` for hyperparameter tuning  
  - `K-Fold Cross-Validation` for robustness  

📊 **Evaluation Metrics:**  
`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`, and `Confusion Matrix`

---

## 🗺️ Susceptibility Mapping  

After model training, the best model is applied to the DEM raster to produce pixel-level susceptibility maps.  

### 🧮 Steps:
1. Extract elevation and slope from DEM.  
2. Use average values for non-topographic features.  
3. Predict pixel-wise probability of landslide occurrence.  
4. Visualize using a color gradient (🟦 Low → 🟥 High Susceptibility).  

---

## 📊 Visualization and Analysis  

The project includes multiple visual components for better interpretation:  

| Visualization | Description |
|---------------|-------------|
| 🧠 **Feature Importance** | Shows contribution of each feature in model decisions |
| 🔥 **Correlation Heatmap** | Visualizes inter-feature relationships |
| 🌄 **Susceptibility Map** | Color-coded map showing landslide risk zones |
| 📉 **Performance Metrics** | Confusion Matrix, F1-score, ROC Curve |

---

## 🧪 LANDSLIDE PREDICTION (LOCALHOST DEPLOYMENT)  

### 🔍 Model Deployment using Pickle  
Below are snapshots of the **Flask-based local deployment** that loads the trained model (`.pkl`) and predicts landslide probability.

<p align="center">
  <img width="800" src="https://github.com/user-attachments/assets/0dd4bab1-8fbc-40e3-be32-68d19d99c862" alt="Prediction UI Screenshot"/>
</p>

<p align="center">
  <img width="800" src="https://github.com/user-attachments/assets/2d32a884-4f4b-4cc7-9282-d51bfc52085f" alt="Prediction Result Screenshot"/>
</p>

---

## 🚀 Getting Started  

Follow these steps to set up the project locally or in Google Colab.  

### 🧰 Prerequisites
- 🐍 Python 3.7+  
- ☁️ Google Colab (recommended) or local Python environment  
- 🧩 Required Python libraries (see `requirements.txt`)  
---

### 💻 Installation  

Clone the repository:
```bash
git clone https://github.com/abhijit826/LANDSLIDE-USING-XG-BOOST-AND-RANDOM-FOREST-CLASSIFIER.git
cd LANDSLIDE-USING-XG-BOOST-AND-RANDOM-FOREST-CLASSIFIER
pip install -r requirements.txt
