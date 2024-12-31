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
    deviation_data[method] = [(e - ex)*100/abs(ex) if e is not None else None for e, ex in zip(enthalpies, enthalpy_expt)]
    print('e:\n', method, enthalpies)
    print(enthalpy_expt)

# Replace missing values with 'Unconv.'
annot_data = deviation_data.map(lambda x: f"{x:.2f}" if pd.notna(x) else "<b style='color:black'>Unconverged</b>")
mask = deviation_data.isna()

# Create a scatter plot
fig = go.Figure()

# Add scatter traces for each DFT method
for method in deviation_data.columns:
    # Handle missing values by masking them
    mask = deviation_data[method].isna()
    fig.add_trace(
        go.Scatter(
            x=n_values,
            y=deviation_data[method],
            mode="markers",
            name=method,
            text=[
                f"Deviation: {val:.2f}%" if pd.notna(val) else "<b style='color:red'>Unconverged</b>"
                for val in deviation_data[method]
            ],
            hoverinfo="text",
            marker=dict(size=8),
            # line=dict(width=2),
            connectgaps=False,   # Do not connect lines across missing data
        )
    )

# Update the layout for better readability
fig.update_layout(width = 800, height = 600,
    title=dict(
        text="Percent Deviation in Reaction Enthalpy_split-valence",
        font=dict(size=20, family="Arial, sans-serif", color="black"),
        x=0.5,
        xanchor="center",  # Anchor to the center
        yanchor="top",  # Anchor to the top
        pad=dict(t=10),  # Adjust the padding above the title
    ),
    margin=dict(
        l=50,  # Left margin
        r=30,  # Right margin
        t=40,  # Top margin, reduce this value to shrink the title area
        b=70,  # Bottom margin
    ),
    xaxis=dict(
        title="Number of Carbon Atoms (n)",
        title_font=dict(size=18, family="Arial, sans-serif", color="blue", ),
        tickfont=dict(size=16, family="Courier New, monospace", color="darkblue"),
        showgrid=True,          # Add gridlines for better visualization
        gridcolor="lightgray",  # Gridline color
        title_standoff=75,      # Increase standoff to move the x-axis title below the legend
    ),
    yaxis=dict(
        title="Percent Deviation (%)",
        title_font=dict(size=18, family="Arial, sans-serif", color="blue"),
        tickfont=dict(size=16, family="Courier New, monospace", color="darkblue"),
        showgrid=True,
        gridcolor="lightgray",
    ),
    font=dict(
        family="Georgia, serif",  # General font for the plot
        size=12,
        color="black",            # General font color
    ),
    legend=dict(
        title=dict(text="DFT Methods<br>", font=dict(size=16, color="purple")),
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,255,255,0.8)",         # Legend background
        bordercolor="rgba(140, 136, 136, 0.8)",  # Legend border color
        borderwidth=1,
        orientation="h",      # Horizontal layout
        xanchor="center",     # Anchor legend to center horizontally
        yanchor="top",        # Anchor legend at the top
        x=0.5,                # Centered horizontally
        y=-0.06,              # Position below the plot
    ),
    plot_bgcolor='white',       # Background color for the plot area
    paper_bgcolor='rgba(248, 244, 244, 0.8)',      # Background color for the entire figure
    hovermode="x unified",      # Unified hover display
    shapes=[
        dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,  # Left edge of the plot area
            y0=0,  # Bottom edge of the plot area
            x1=1,  # Right edge of the plot area
            y1=1,  # Top edge of the plot area
            line=dict(
                color="darkgray",  # Border color
                width=0.8,        # Border width
            ),
            layer="below",      # Draw behind other plot elements
        )
    ]
)


# Show the plot
fig.show()

# Save the figure as PNG
fig.write_image("percent_deviation_enthalpy_split.png", format="png", scale=5)


# save a html
import plotly.io as pio
pio.write_html(fig, 'percent_deviation_split_enthalpy.html', auto_open=False, )