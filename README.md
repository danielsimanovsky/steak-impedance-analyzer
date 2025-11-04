# 🔬 Impedance Analysis Dashboard

This is a Streamlit web application for converting, analyzing, and visualizing impedance spectroscopy (`.spec`) data.

It allows a user to select multiple experiment folders, automatically convert the raw data, and generate a series of 2D and 3D plots to analyze the results.

---

## Features

* **Multi-Experiment Analysis:** Select and analyze multiple experiment folders (e.g., `1cm_...`, `2cm_...`) at once.
* **Automatic Data Conversion:** Converts raw `.spec` files into sequentially named `.csv` files (e.g., `0.0_nf.csv`, `0.1_nf.csv`...) based on their folder's sort order.
* **Data Cleaning:** The conversion step automatically cleans up old/badly named `*_nf.csv` files to prevent data contamination.
* **2D Plots (Per-Experiment):** Generates two 2D scatter plots for *each* experiment folder:
    1.  Impedance |Z| vs. Frequency (with a legend for each capacitance).
    2.  Minimum |Z| vs. Capacitance.
* **3D Plots (Combined):** Generates interactive 3D plots that combine data from *all* selected experiments:
    1.  3D Spectra vs. Distance.
    2.  3D Minimum |Z| vs. Capacitance vs. Distance.
* **Summary Plots (Combined):** Generates 2D summary plots for all selected data:
    1.  Minimum |Z| vs. Relative Distance.
    2.  $\Delta$|Z| (Impedance Range) vs. Relative Distance.

---

## 💻 Installation & Setup

### 1. Requirements

* Python 3.8+
* The libraries listed in `requirements.txt`.

### 2. Setup Instructions

1.  Place the project files (`app.py`, `analysis.py`, `requirements.txt`) in a folder.
2.  Open your terminal (CMD, PowerShell, etc.).
3.  Navigate to the project folder:
    ```bash
    cd path\to\your\project_folder
    ```
4.  (Optional but Recommended) Create a virtual environment:
    ```bash
    py -m venv .venv
    ```
5.  Activate the environment:
    ```bash
    .\.venv\Scripts\activate
    ```
6.  Install all required libraries at once:
    ```bash
    py -m pip install -r requirements.txt
    ```

---

## 🚀 How to Use the App

1.  Run the app from your terminal:
    ```bash
    py -m streamlit run app.py
    ```
2.  Your web browser will open with the app interface.
3.  **In the sidebar, follow these steps:**
    * **Step 1: Setup:** Paste the full, absolute paths to your **Root Data Folder** and your desired **Output Folder**.
    * **Step 2: Configuration:**
        * Select all the experiment folders you want to analyze (e.g., `1cm_100uApp`, `2cm_100uApp`).
        * Set the **Steak Size (cm)** for relative distance calculations.
        * Set the **Z-Limit (Ohm)** to filter out high-impedance noise if needed (0 = no limit).
    * **Step 3: Analysis Steps:**
        * **IMPORTANT:** The first time you run, or if you add new `.spec` data, you **must check the "Run .spec to .csv Conversion" box.** This cleans old files and creates the new, correctly named CSVs.
        * Select which 2D and 3D plots you want to generate.
4.  Click the **"Run Analysis"** button.
5.  View the results in the main window.

---

## 📁 Required Data Structure

The app requires a specific folder structure to work. You must provide the path to the **Root Data Folder** (the one containing all your `...cm...` folders).

### Naming Scheme & Structure

* **Root Data Folder:** The folder you select in the UI.
* **Experiment Folder:** A folder for each distance, e.g., `1cm_100uApp`.
* **Timestamp Folder:** Inside the experiment folder, you must have one subfolder for each capacitance measurement. The script **sorts these folders by name** to determine the capacitance order.
* **Raw File:** The `.spec` file itself, which can have any name (e.g., `measurement config_00001.spec`).