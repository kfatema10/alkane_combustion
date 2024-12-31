"""
Script: heatmap_enthalpy_deviation.py

Author: Kaniz Fatema  
Email: fatemache10@gmail.com, kaniz.fatema@stonybrook.edu  

Date Created: November 27, 2024  
Last Updated: December 17, 2024 

Description:
    This script generates an interactive heatmap to visualize enthalpy deviations 
    from experimental values for hydrocarbon combustion reactions using multiple DFT methods. 
    The script processes computational data stored in CSV files and compares it to experimental values.

Features:
    - Reads thermodynamic data (reaction enthalpy) from multiple computational methods.
    - Calculates deviations between computational and experimental enthalpy values.
    - Highlights missing or unconverged data as "Unconverged" in the heatmap.
    - Produces an interactive Plotly heatmap with:
        - Configurable font sizes and color scales.
        - Annotations for data points with improved readability.
    - Saves the high-resolution heatmap as a PNG image. 

Dependencies:
    - pandas: For reading and processing CSV data.
    - numpy: For numerical operations.
    - plotly: For generating interactive heatmaps.
    - pathlib: For dynamic file handling.

Usage:
    - Place the thermodynamic data files (reaction_parameters_*.csv) in the working directory.
    - Run the script to generate an interactive heatmap and save it as a PNG file.

Notes:
    - Experimental data is hardcoded for enthalpy deviations.
    - The output image is saved with a high DPI using `scale=3` for better resolution.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Load the experimental enthalpy data
enthalpy_expt = [-212.954, -373.088, -530.5927, -687.8585, -838.6711,
                 -994.9809, -1151.291, -1307.361, -1463.91, -1619.981]

# Define folder path and method to extract filenames
folder_path = Path('./')
filenames = sorted(folder_path.glob('*.csv'))

# Extract method names and initialize data
method_names = [f"{str(file.stem).split('_')[-2]}_{str(file.stem).split('_')[-1]}" for file in filenames]
n_values = list(range(1, 11))  # n = 1 to 10

# Prepare a DataFrame to store enthalpy deviations
deviation_data = pd.DataFrame(index=n_values, columns=method_names)

# Process CSV files to calculate deviations
for file, method in zip(filenames, method_names):
    df = pd.read_csv(file)
    enthalpies = [None] * len(n_values)
    for _, row in df.iterrows():
        n = int(row["No. of C atom"])
        enthalpies[n - 1] = row["Reaction Enthalpy (kcal/mol)"]
    deviation_data[method] = [e - ex if e is not None else None for e, ex in zip(enthalpies, enthalpy_expt)]

# Replace missing values with 'Unconv.'
annot_data = deviation_data.map(lambda x: f"{x:.2f}" if pd.notna(x) else "<b style='color:black'>Unconverged</b>")
mask = deviation_data.isna()

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=deviation_data.values,
    x=method_names,  # Use full method names
    y=[f"n={i}" for i in n_values],
    text=annot_data.values,
    texttemplate="%{text}",
    textfont=dict(size=12),  # modify font size for data points
    colorscale="magma_r",  # magma_r
    # colorscale = [[0, 'hsl(60, 100.00%, 51.60%)'], [1, 'rgba(245, 34, 34, 0.81)']],
    colorbar=dict(title="Deviation<br>(kcal/mol)")
))

fig.update_layout(
            title=dict(text="Deviations in Enthalpy (Computational - Experimental)_correlation-consistent",
                font=dict(size=16, color="darkblue"),
                x=0.50,  # Center alignment
                y=0.95,  # Title padding 
                ),
            xaxis=dict(title=dict(text="DFT Methods",
                        font=dict(size=16, color="black"), ),
                tickangle=10,  # Rotate x-ticks 
                tickfont=dict(size=14, color="black")
            ),
            yaxis=dict(
                title=dict(text="Number of Carbon Atoms (n)",
                            font=dict(size=16, color="black"), ),
                            tickangle=0,  # Rotate x-ticks 
                            tickfont=dict(size=14, color="black")
            ),
            height=900,
            width=1000,  
            plot_bgcolor='white',  # Background color for the plot area
            paper_bgcolor='lightgray',  # Background color for the entire figure
            )


# Show the heatmap
fig.show()

# Save the figure as PNG
fig.write_image("heatmap_enthalpy_cc.png", format="png", scale=5)


# save a html
import plotly.io as pio
pio.write_html(fig, 'heatmap_cc_enthalpy.html', auto_open=False, )