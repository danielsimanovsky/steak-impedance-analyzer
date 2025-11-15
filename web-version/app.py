import streamlit as st
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple

# --- New libraries for unzipping in memory ---
import zipfile
import tempfile

# Import your backend class
try:
    from analysis import ImpedanceAnalyzer
except ImportError:
    st.error("Missing 'analysis.py'. Please make sure it's in the same folder.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Impedance Analysis Dashboard",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Impedance Analysis Dashboard (Web Version)")
st.markdown("Upload a single `.zip` file containing your data folders to begin.")


# ---
# 3D PLOTTING FUNCTIONS
# (Now updated to accept plot_type)
# ---

def get_all_csv_data(root_dir: Path, selected_folders: List[str],
                     z_limit: Optional[float], coil_inductance: float
                     ) -> pd.DataFrame:
    """
    Helper function to loop through all selected folders and compile
    all CSV data into one master DataFrame for 3D plotting.
    """
    all_data = []

    # --- CONVERSION UPDATE: nH to H ---
    L_henry = coil_inductance * 1e-9  # Convert L from nH to H

    for folder_name in selected_folders:
        exp_path = root_dir / folder_name

        try:
            distance_val = float(folder_name.split("cm")[0])
        except Exception:
            print(f"Could not parse distance from: {folder_name}")
            continue

        csv_files = list(exp_path.rglob("*_nf.csv"))  # Only find the converted files

        for file in csv_files:
            try:
                cap_val = float(file.name.split("_")[0])
            except Exception:
                continue  # Skip non-data CSVs

            try:
                df = pd.read_csv(file, skiprows=7)
                df['distance'] = distance_val
                df['capacitance'] = cap_val

                # --- NEW: Add all data types ---
                df['real_z'] = df["Re[Ohm]"]
                df['imag_z'] = df["Im[Ohm]"]
                df['magnitude'] = np.sqrt(df["Re[Ohm]"] ** 2 + df["Im[Ohm]"] ** 2)

                # --- START OF NEW |Z| (Parallel) CALCULATION ---
                # --- CONVERSION UPDATE: nF to F ---
                C_farad = df['capacitance'] * 1e-9  # Convert C from nF to F

                with np.errstate(divide='ignore', invalid='ignore'):
                    XL = 2 * np.pi * df['frequency[Hz]'] * L_henry
                    XC = 1 / (2 * np.pi * df['frequency[Hz]'] * C_farad)

                    term1 = (1 / df['real_z']) ** 2
                    term2 = (1 / XL - 1 / XC) ** 2

                    df['parallel_z'] = 1 / np.sqrt(term1 + term2)

                df['parallel_z'] = df['parallel_z'].replace([np.inf, -np.inf], np.nan)
                # --- END OF NEW CALCULATION ---

                if z_limit:
                    # Filter based on magnitude, regardless of plot type
                    df = df[df['magnitude'] <= z_limit].copy()

                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file.name}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def plot_3d_distance_plotly(all_data: pd.DataFrame, y_column: str, y_label: str) -> go.Figure:
    """Creates an interactive 3D plot of Freq vs. Distance vs. [Selected Value]."""

    fig = go.Figure()

    # Use a smaller subset if data is too large (optional, for performance)
    plot_data = all_data.sample(min(len(all_data), 50000))

    fig.add_trace(go.Scatter3d(
        x=plot_data['frequency[Hz]'],
        y=plot_data['distance'],
        z=plot_data[y_column],
        mode="markers",
        marker=dict(size=2, opacity=0.5, color=plot_data[y_column], colorscale='Viridis'),
        name=y_label
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Frequency (Hz)", type="log"),
            # --- UPDATED: Also log scale parallel Z ---
            zaxis=dict(title=y_label, type="log" if y_column in ['magnitude', 'parallel_z'] else 'linear'),
            yaxis=dict(title="Relative distance (cm)")
        ),
        title=f"3D {y_label} vs. Frequency vs. Distance",
        legend=dict(itemsizing="constant", font=dict(size=8)),
    )
    return fig


def plot_min_impedance_vs_distance(all_data: pd.DataFrame, steak_size: float,
                                   y_column: str, y_label: str
                                   ) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plots the minimum [Selected Value] found for each distance."""

    # Find the minimum of the selected column for *each distance folder*
    min_z_by_dist = all_data.groupby('distance')[y_column].min()

    # Find the frequency at that minimum
    min_freq_by_dist = all_data.loc[all_data.groupby('distance')[y_column].idxmin()] \
        .set_index('distance')['frequency[Hz]']

    df_results = pd.DataFrame({
        "Distance (cm)": min_z_by_dist.index,
        f"Min {y_label}": min_z_by_dist.values,
        "Frequency at Min (Hz)": min_freq_by_dist.loc[min_z_by_dist.index].values
    })

    df_results["Relative Distance"] = df_results["Distance (cm)"] / steak_size
    df_results = df_results.sort_values(by="Relative Distance")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(df_results["Relative Distance"], df_results[f"Min {y_label}"],
            marker="o", linestyle="-", color="b")
    ax.set_xlabel("Relative Distance (Dcap/Ddrive)", fontsize=12)
    ax.set_ylabel(f"Minimum {y_label}", fontsize=12)
    ax.set_title(f"Minimum {y_label} vs Distance", fontsize=14, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    return (fig, df_results)


def plot_impedance_range_vs_distance(all_data: pd.DataFrame, steak_size: float,
                                     y_column: str, y_label: str
                                     ) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plots the [Selected Value] range (delta) for each distance."""

    grouped = all_data.groupby('distance')[y_column]
    min_z = grouped.min()
    max_z = grouped.max()

    df_results = pd.DataFrame({
        "Distance (cm)": min_z.index,
        f"Min {y_label}": min_z.values,
        f"Max {y_label}": max_z.loc[min_z.index].values,
    })
    df_results[f"Δ{y_label}"] = df_results[f"Max {y_label}"] - df_results[f"Min {y_label}"]
    df_results["Relative Distance"] = df_results["Distance (cm)"] / steak_size
    df_results = df_results.sort_values(by="Relative Distance")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(df_results["Relative Distance"], df_results[f"Δ{y_label}"],
            marker="o", linestyle="-", color="r")
    ax.set_xlabel("Relative distance (Dcap/Ddrive)", fontsize=12)
    ax.set_ylabel(f"Δ{y_label}", fontsize=12)
    ax.set_title(f"{y_label} Range vs Distance", fontsize=14, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    return (fig, df_results)


def plot_3d_impedance_vs_capacitance(all_data: pd.DataFrame, steak_size: float,
                                     y_column: str, y_label: str
                                     ) -> Tuple[go.Figure, pd.DataFrame]:
    """Plots an interactive 3D graph of [Selected Value] at Min |Z| vs. Cap vs. Dist."""

    # --- IMPORTANT: This logic is different ---
    # We find the MINIMUM *MAGNITUDE* (resonance)
    min_mag_data = all_data.loc[all_data.groupby(['distance', 'capacitance'])['magnitude'].idxmin()]
    # But we will plot the value of the USER-SELECTED COLUMN

    min_mag_data['Relative Distance'] = min_mag_data['distance'] / steak_size

    fig = go.Figure()

    for dist, group in min_mag_data.groupby('Relative Distance'):
        group = group.sort_values(by='capacitance')

        fig.add_trace(go.Scatter3d(
            x=group['capacitance'],
            y=group['Relative Distance'],
            z=group[y_column],  # Plot the selected value
            mode="lines+markers", marker=dict(size=4), line=dict(width=2),
            name=f"Rel. Dist: {dist:.2f}",
            # --- Label Update: pF -> nF ---
            hovertemplate=f"Rel. Dist: %{{y:.2f}}<br>{y_label}: %{{z:.2f}} Ω<br>Capacitance: %{{x}} nF<extra></extra>"
        ))

    fig.update_layout(
        scene=dict(
            # --- Label Update: pF -> nF ---
            xaxis=dict(title="Capacitance (nF)"),
            # --- UPDATED: Also log scale parallel Z ---
            zaxis=dict(title=f"{y_label} at Resonance",
                       type="log" if y_column in ['magnitude', 'parallel_z'] else 'linear'),
            yaxis=dict(title="Relative Distance (Dcap/Ddrive)")
        ),
        title=f"3D {y_label} at Resonance vs. Capacitance vs. Distance"
    )

    df_results = min_mag_data[[
        "Relative Distance", "capacitance", y_column
    ]].rename(columns={
        # --- Label Update: pF -> nF ---
        "capacitance": "Capacitance (nF)",
        y_column: f"{y_label} at Resonance (Ω)"
    })

    return (fig, df_results)


# ---
# START OF NEW STREAMLIT UI (WEB VERSION)
# ---

# --- 1. Sidebar Configuration ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your Root Data Folder as a .zip file",
        type="zip"
    )

    st.header("2. Configuration")

    st.subheader("Capacitance Settings")
    # --- Label Update: pF -> nF ---
    cap_start = st.number_input("Capacitance Start (nF)", value=0.0, step=0.1, format="%.2f")
    cap_step = st.number_input("Capacitance Step (nF)", value=0.1, step=0.1, format="%.2f")

    st.subheader("Physical Settings")
    steak_size = st.number_input(
        "Steak Size (cm)",
        min_value=0.1, value=24.0, step=0.1
    )
    # --- Label Update: uH -> nH ---
    coil_inductance = st.number_input("Coil Inductance (nH)", value=0.0, step=0.1)

    st.subheader("Analysis Settings")
    z_limit = st.number_input(
        "Z-Limit (Ohm) (0 for no limit)",
        min_value=0, value=100000, step=1000
    )
    z_limit_val = z_limit if z_limit > 0 else None

    # --- New Plot Type Selector ---
    plot_type_str = st.radio(
        "Select Data to Plot:",
        ('|Z| (Magnitude)', '|Z| (Parallel Model)', 'Re(Z) (Real)', 'Im(Z) (Imaginary)'),
        index=0
    )

    # --- START OF FIX ---
    # Map the friendly string to the column name AND a safe filename
    plot_type_map = {
        '|Z| (Magnitude)': ('magnitude', '|Z| (Ω)', 'MagZ'),
        '|Z| (Parallel Model)': ('parallel_z', '|Z| Parallel (Ω)', 'MagZ_Parallel'),
        'Re(Z) (Real)': ('real_z', 'Re(Z) (Ω)', 'ReZ'),
        'Im(Z) (Imaginary)': ('imag_z', 'Im(Z) (Ω)', 'ImZ')
    }
    plot_y_column, plot_y_label, plot_type_2d = plot_type_map[plot_type_str]
    # --- END OF FIX ---

    st.header("3. Analysis Steps")
    st.info("Conversion is automatic on every run.")

    run_2d_plots = st.checkbox("2D Plots (per Experiment)", value=True)
    run_3d_dist = st.checkbox("3D Spectra vs. Distance", value=True)
    run_3d_cap = st.checkbox("3D |Z| vs. Capacitance", value=True)
    run_min_z = st.checkbox("Min Value vs. Distance", value=True)
    run_delta_z = st.checkbox("Value Range vs. Distance", value=True)

    run_button = st.button(f"**🚀 Run Analysis**")

# --- 4. Main Analysis Area ---
if not uploaded_file:
    st.info("Please upload your data .zip file using the sidebar to begin.")
    st.stop()

if run_button:
    # --- This 'with' block creates a secure, temporary folder ---
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # 1. Unzip the file
        with st.spinner("Extracting data..."):
            try:
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir_path)
            except Exception as e:
                st.error(f"Error extracting ZIP file: {e}")
                st.stop()

        # 2. Find the unzipped root folder
        unzipped_items = list(temp_dir_path.iterdir())
        if len(unzipped_items) == 1 and unzipped_items[0].is_dir():
            root_dir = unzipped_items[0]
        else:
            root_dir = temp_dir_path

        st.success(f"Data extracted. Found root folder: `{root_dir.name}`")

        # 3. Find the experiment folders inside
        try:
            exp_folders = [f for f in root_dir.iterdir() if f.is_dir() and f.name[0].isdigit()]
            selected_experiments = sorted([f.name for f in exp_folders])
            if not selected_experiments:
                st.error(f"No experiment folders (e.g., '1cm_...') found in the zip file.")
                st.stop()
        except Exception as e:
            st.error(f"Could not read experiment folders from zip: {e}")
            st.stop()

        # --- Step 1: Conversion & 2D Plots (Loop) ---
        st.header(f"1. 2D Analysis (Plotting {plot_type_str})")

        for exp_name in selected_experiments:
            with st.expander(f"▼ Results for: {exp_name}", expanded=True):

                exp_path = root_dir / exp_name
                exp_output_path = temp_dir_path / "results" / exp_name

                st.markdown(f"**Analyzing:** `{exp_name}`")

                try:
                    analyzer = ImpedanceAnalyzer(
                        experiment_dir=exp_path,
                        output_dir=exp_output_path,
                        steak_size=steak_size,
                        coil_inductance=coil_inductance,
                        z_limit=z_limit_val
                    )

                    # --- Run conversion (this is now mandatory) ---
                    with st.spinner(f"[{exp_name}] Converting .spec to .csv..."):
                        status_msg = analyzer.run_spec_to_csv_conversion(cap_start, cap_step)
                        st.write(f"Conversion complete: {status_msg}")

                    # --- Run 2D plots ---
                    if run_2d_plots:
                        st.subheader(f"2D Plots for {exp_name}")
                        with st.spinner(f"[{exp_name}] Generating 2D plots..."):
                            fig_z_freq, fig_z_cap = analyzer.plot_2d_graphs(plot_type_2d)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(fig_z_freq)
                            with col2:
                                st.pyplot(fig_z_cap)
                            plt.close('all')

                except Exception as e:
                    st.error(f"Failed to analyze {exp_name}: {e}")
                    continue

        st.success("2D analysis complete.")
        st.markdown("---")

        # --- Step 2: 3D Plots (Combined) ---
        st.header(f"2. 3D & Summary Analysis (Plotting {plot_type_str})")

        all_data_df = pd.DataFrame()
        if any([run_3d_dist, run_3d_cap, run_min_z, run_delta_z]):
            with st.spinner("Loading all data for 3D/Summary plots..."):
                try:
                    all_data_df = get_all_csv_data(
                        root_dir, selected_experiments, z_limit_val, coil_inductance
                    )
                    if all_data_df.empty:
                        st.error("No data found for 3D/Summary plots.")
                    else:
                        st.success(f"Loaded {len(all_data_df)} total data points.")
                except Exception as e:
                    st.error(f"Failed to load combined data: {e}")

        if not all_data_df.empty:
            if run_3d_dist:
                st.subheader(f"3D {plot_type_str} vs. Distance")
                with st.spinner("Generating 3D distance plot..."):
                    try:
                        fig = plot_3d_distance_plotly(all_data_df, plot_y_column, plot_y_label)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate 3D distance plot: {e}")

            if run_min_z:
                st.subheader(f"Minimum {plot_type_str} vs. Distance")
                with st.spinner(f"Generating Min {plot_type_str} plot..."):
                    try:
                        fig, df = plot_min_impedance_vs_distance(all_data_df, steak_size,
                                                                 plot_y_column, plot_y_label)
                        st.pyplot(fig)
                        with st.expander("View Data"):
                            st.dataframe(df)
                    except Exception as e:
                        st.error(f"Could not generate Min {plot_type_str} plot: {e}")

            if run_delta_z:
                st.subheader(f"{plot_type_str} Range (Δ) vs. Distance")
                with st.spinner(f"Generating Δ{plot_type_str} plot..."):
                    try:
                        fig, df = plot_impedance_range_vs_distance(all_data_df, steak_size,
                                                                   plot_y_column, plot_y_label)
                        st.pyplot(fig)
                        with st.expander("View Data"):
                            st.dataframe(df)
                    except Exception as e:
                        st.error(f"Could not generate Δ{plot_type_str} plot: {e}")

            if run_3d_cap:
                st.subheader(f"3D {plot_type_str} at Resonance vs. Capacitance")
                with st.spinner(f"Generating 3D capacitance plot..."):
                    try:
                        fig, df = plot_3d_impedance_vs_capacitance(all_data_df, steak_size,
                                                                   plot_y_column, plot_y_label)
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("View Raw Data"):
                            st.dataframe(df)
                    except Exception as e:
                        st.error(f"Could not generate 3D capacitance plot: {e}")

    st.success(f"🎉 **Full analysis complete!**")
    st.balloons()