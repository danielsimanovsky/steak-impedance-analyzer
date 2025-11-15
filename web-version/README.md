🔬 Impedance Analyzer Dashboard

This repository contains two versions of a Streamlit app for analyzing impedance spectroscopy (.spec) data.

Web Version (Recommended): A hosted app that runs on Streamlit Community Cloud. You upload a .zip file of your data.

Local Version: A local tool that runs on your computer and reads data directly from your hard drive.

Both versions perform the same analysis, including:

Automatic .spec to .csv conversion.

Selectable plotting for |Z|, Re(Z), Im(Z), and a calculated |Z| (Parallel Model).

Customizable capacitance start/step and coil inductance (in nF/nH).

2D plots per experiment and combined 3D/Summary plots.

📁 Repository Structure

/
├── requirements.txt    (Libraries for both versions)
├── README.md           (This file)
│
├── local-version/      (Runs on your local PC, reads local paths)
│   ├── app.py
│   └── analysis.py
│
└── web-version/        (Runs on Streamlit Cloud, uses .zip upload)
    ├── app.py
    └── analysis.py


🚀 Web Version (Recommended)

This version is hosted online and accessible via a URL. It requires you to upload your data as a single .zip file.

How to Deploy Your Own

Fork/Clone this repository to your own GitHub account.

Sign up for Streamlit Community Cloud with your GitHub account.

Click "New app" and select your forked repository.

Set the "Main file path" to web-version/app.py.

Click Deploy.

How to Use the Web App

Prepare Your Data: Create a single .zip file of your root data folder (see Data Structure below).

Open the App: Go to your new Streamlit URL.

Upload: In the sidebar, click "Browse files" and upload your .zip file.

Configure:

Set the Capacitance Start (nF) and Step (nF) (e.g., 0.0 and 0.1).

Set the Steak Size (cm) and Coil Inductance (nH).

Select the data you wish to plot (e.g., |Z| (Parallel Model)).

Run: Click the "Run Analysis" button.

💻 Local-Only Version

This version runs on your computer and reads files directly from your hard drive.

How to Run

Clone this repository to your computer.

Open your terminal (CMD, PowerShell, etc.).

Navigate to the local-version folder:

cd path\to\repository\local-version


(Recommended) Create & activate a virtual environment:

# Create
py -m venv .venv
# Activate
.\.venv\Scripts\activate


Install libraries from the root requirements.txt file:

py -m pip install -r ../requirements.txt


Run the app:

py -m streamlit run app.py


The app will open in your browser, asking you for local folder paths.

📦 Required Data Structure

Both versions of the app require the same data structure.

1. Folder Structure

The script expects a root folder containing one subfolder for each experiment (distance). Inside each experiment folder, it expects one subfolder for each measurement, sorted by name (e.g., by timestamp).

Root_Data_Folder/              <-- This is the folder you zip
│
├── 1cm_100uApp/
│   │
│   ├── 20250916 14.52.30/     <-- Timestamped folder (for 0.0 nF)
│   │   └── measurement config_00001.spec
│   │
│   ├── 20250916 14.53.00/     <-- Timestamped folder (for 0.1 nF)
│   │   └── measurement config_00001.spec
│   │
│   └── ... (and so on)
│
├── 2cm_100uApp/
│   │
│   ├── 20250916 15.06.29/
│   │   └── measurement config_00001.spec
│   │
│   └── ...
│
└── ... (etc.)


2. For the Web App (.zip file)

To create the .zip file for the web app:

Find your Root_Data_Folder.

Right-click on this single folder.

Select "Send to" -> "Compressed (zipped) folder".

Upload the resulting .zip file.