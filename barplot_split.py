import pandas as pd
# import plotly.graph_objects as go
from pathlib import Path
import plotly.express as px

# Define folder path and method to extract filenames
folder_path = Path('./')
filenames = sorted(folder_path.glob('*.csv'))

# Experimental enthalpy values (kcal/mol) for n=1 to n=10
enthalpy_expt = [-212.954, -373.088, -530.5927, -687.8585, -838.6711,
                 -994.9809, -1151.291, -1307.361, -1463.91, -1619.981]

# Extract method names and data, last 2 words give method name as 'functional_basis set' 
method_names = []
for file in filenames:
    stem_parts = str(file.stem).split('_')  # Split the filename into parts
    method_name = f"{stem_parts[-2]}_{stem_parts[-1]}"  # Combine the last two parts
    method_names.append(method_name)

# Initialize lists for plotting
n_values = list(range(1, 11))  # n=1 to n=10
enthalpy_data = {method: [None] * len(n_values) for method in method_names}

# Process CSV files
for file, method in zip(filenames, method_names):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        n = int(row['No. of C atom'])
        enthalpy_data[method][n - 1] = row['Reaction Enthalpy (kcal/mol)']

# Identify missing n-values dynamically
missing_n = []
for method, enthalpies in enthalpy_data.items():
    for i, value in enumerate(enthalpies):
        if value is None:
            missing_n.append(n_values[i])

# Deduplicate and format missing n-values
missing_n = sorted(set(missing_n))
missing_message = f"Note: Computational method for n={', '.join(
                        map(str, missing_n))}<br>was not converged for some methods."

# Combine enthalpy data into a single DataFrame for plotting
bar_data = []
for method, enthalpies in enthalpy_data.items():
    for i, value in enumerate(enthalpies):
        if value is not None:  # Skip missing values
            bar_data.append({'Method': method, 'n': f"n={i + 1}", 'Enthalpy': value})

# Add experimental values to the bar data
for i, value in enumerate(enthalpy_expt):
    bar_data.append({'Method': 'Experimental', 'n': f"n={i + 1}", 'Enthalpy': value})

# Convert to DataFrame
bar_df = pd.DataFrame(bar_data)

# Generate dynamic colors using seaborn or matplotlib
unique_methods = bar_df['Method'].unique()

# Define a list of colors (make sure it's long enough for all methods)
predefined_colors = ['#4575b4', '#91bfdb', '#e0f3f8', 
                        '#ffffbf', '#fee090', '#fc8d59', '#d73027']
                        
# Assign colors to methods
dynamic_colors = {}
for i, method in enumerate(unique_methods):
    dynamic_colors[method] = predefined_colors[i % len(predefined_colors)]  # Cycle through colors if methods > colors

# Create a grouped bar plot
fig_enthalpy = px.bar(
    bar_df,
    x="Enthalpy", y="n",
    color="Method",
    barmode="group",
    template="simple_white",
    color_discrete_map=dynamic_colors,  # Apply custom colors
    
)

# Add a custom message dynamically at the bottom-left corner
fig_enthalpy.add_annotation(
    text=missing_message,
    x=min(enthalpies)+100 , y=min(n_values)+2,  # Adjust dynamically to bottom-left
    showarrow=False,
    font=dict(size=16, color="black"),
    align="left",
    bgcolor='rgba(255, 255, 255, 0.8)',
    bordercolor='rgba(202, 193, 193, 0.8)',
)

# Customize layout for better visibility
fig_enthalpy.update_layout(
    height=900, width=1000,
    bargap=0.3,
    title=dict(
        text="Comparison of Reaction Enthalpy Across DFT Methods and<br>Experimental Data_split-valence",
        font=dict(size=22, family="Arial, sans-serif", color="black"),
        x=0.5, y=0.97,  # Center the title
        # xanchor='center'
    ),
    xaxis=dict(
        title="Reaction Enthalpy (kcal/mol)",
        titlefont=dict(size=18, family="Arial, sans-serif", color="black"),
        tickfont=dict(size=16, family="Arial, sans-serif"),
        range=(0,-1700),
    ),
    yaxis=dict(
        title="Number of Carbon Atoms (n)",
        titlefont=dict(size=18, family="Arial, sans-serif", color="black"),
        tickfont=dict(size=16, family="Arial, sans-serif"),
    ),
    legend=dict(
        title="Method",
        title_font=dict(size=18, family="Arial, sans-serif"),
        font=dict(size=14, family="Arial, sans-serif"),
        x=0.615, y=0.05,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(202, 193, 193, 0.8)',
        borderwidth=1,
    ),
    plot_bgcolor='white',       # Background color for the plot area
    paper_bgcolor='rgba(248, 244, 244, 0.8)',      # Background color for the entire figure
    hovermode="x unified",      # Unified hover display
)

# Show the bar plot
fig_enthalpy.show()

# Save as PNG
fig_enthalpy.write_image("bar_plot_split_enthalpy.png", format="png", scale=5)

# save a html
import plotly.io as pio
pio.write_html(fig_enthalpy, 'bar_plot_split_enthalpy.html', auto_open=False, )

# ------------------------------------------
# bar plot for Gibbs energy

# Initialize lists for plotting
n_values = list(range(1, 11))  # n=1 to n=10
gibbs_data = {method: [None] * len(n_values) for method in method_names}

# Process CSV files
for file, method in zip(filenames, method_names):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        n = int(row['No. of C atom'])
        gibbs_data[method][n - 1] = row['Gibbs Free Energy Change (kcal/mol)']

# Combine enthalpy data into a single DataFrame for plotting
bar_data = []
for method, gibbs in gibbs_data.items():
    for i, value in enumerate(gibbs):
        if value is not None:  # Skip missing values
            bar_data.append({'Method': method, 'n': f"n={i + 1}", 'Gibbs': value})

# Convert to DataFrame
bar_df = pd.DataFrame(bar_data)

# Generate dynamic colors using seaborn or matplotlib
unique_methods = bar_df['Method'].unique()

# Generate colors programmatically using HSV
import colorsys

dynamic_colors = {}
num_methods = len(unique_methods)
for i, method in enumerate(unique_methods):
    hue = i / num_methods  # Distribute hues evenly in the range [0, 1]
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)  # High saturation and value for vibrancy
    color = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
    dynamic_colors[method] = color

# Generate dynamic colors using seaborn or matplotlib
unique_methods = bar_df['Method'].unique()

# Define a list of colors (make sure it's long enough for all methods)
predefined_colors = ['#4d4d4d', '#999999', '#e0e0e0', 
                        '#fddbc7', '#ef8a62', '#b2182b']

# Assign colors to methods
dynamic_colors = {}
for i, method in enumerate(unique_methods):
    dynamic_colors[method] = predefined_colors[i % len(predefined_colors)]  # Cycle through colors if methods > colors

# Create a grouped bar plot
fig_gibbs = px.bar(
    bar_df,
    x="Gibbs", y="n",
    color="Method",
    barmode="group",
    template="simple_white",
    color_discrete_map=dynamic_colors,  # Apply custom colors
    
)

# Add a custom message dynamically at the bottom-left corner
fig_gibbs.add_annotation(
    text=missing_message,
    x=min(gibbs)+300 , y=min(n_values)+2,  # Adjust dynamically to bottom-left
    showarrow=False,
    font=dict(size=16, color="black"),
    align="left",
    bgcolor='rgba(255, 255, 255, 0.8)',
    bordercolor='rgba(202, 193, 193, 0.8)',
)

# Customize layout for better visibility
fig_gibbs.update_layout(
    height=900, width=1000,
    bargap=0.3,
    title=dict(
        text="Comparison of Change in Gibbs Free Energy Across DFT Methods_split-valence",
        font=dict(size=22, family="Arial, sans-serif", color="black"),
        x=0.5, y=0.97,  # Center the title
        # xanchor='center'
    ),
    xaxis=dict(
        title="Change in Gibbs Free Energy (kcal/mol)",
        titlefont=dict(size=18, family="Arial, sans-serif", color="black"),
        tickfont=dict(size=16, family="Arial, sans-serif"),
        range=(0,-1500),
    ),
    yaxis=dict(
        title="Number of Carbon Atoms (n)",
        titlefont=dict(size=18, family="Arial, sans-serif", color="black"),
        tickfont=dict(size=16, family="Arial, sans-serif"),
    ),
    legend=dict(
        title="Method",
        title_font=dict(size=18, family="Arial, sans-serif"),
        font=dict(size=14, family="Arial, sans-serif"),
        x=0.56, y=0.05,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(202, 193, 193, 0.8)',
        borderwidth=1,
    ),
    plot_bgcolor='white',       # Background color for the plot area
    paper_bgcolor='rgba(248, 244, 244, 0.8)',      # Background color for the entire figure
    hovermode="x unified",      # Unified hover display
)

# Show the bar plot
fig_gibbs.show()

# Save as PNG
fig_gibbs.write_image("bar_plot_split_gibbs.png", format="png", scale=5)

# save a html
import plotly.io as pio
pio.write_html(fig_gibbs, 'bar_plot_split_gibbs.html', auto_open=False, )

# ------------------------------------------
# bar plot for entropy
# Initialize lists for entropy plotting
entropy_data = []

# Compute entropy for each method and n
for method in method_names:
    for i in range(len(n_values)):
        enthalpy = enthalpy_data[method][i]
        gibbs = gibbs_data[method][i]
        
        if enthalpy is not None and gibbs is not None:
            # Calculate entropy using ΔS = (ΔH - ΔG) / T
            delta_s = (enthalpy - gibbs) / 298.15  # T = 298.15 K (room temperature)
            entropy_data.append({'Method': method, 'n': f"n={i + 1}", 'Entropy': delta_s})

# Convert to DataFrame
entropy_df = pd.DataFrame(entropy_data)

# Generate colors programmatically (use HSV or any preferred palette)
dynamic_colors = {}
unique_methods = entropy_df['Method'].unique()
num_methods = len(unique_methods)

for i, method in enumerate(unique_methods):
    hue = i / num_methods  # Distribute hues evenly in the range [0, 1]
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)  # High saturation and value for vibrancy
    color = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
    dynamic_colors[method] = color

# Generate dynamic colors using seaborn or matplotlib
unique_methods = bar_df['Method'].unique()

# Define a list of colors (make sure it's long enough for all methods)
predefined_colors = ['#b0b0ac', '#fed976', '#feb24c', 
                        '#fd8d3c', '#f03b20', '#bd0026']
                        
# Assign colors to methods
dynamic_colors = {}
for i, method in enumerate(unique_methods):
    dynamic_colors[method] = predefined_colors[i % len(predefined_colors)]  # Cycle through colors if methods > colors

# Create a grouped bar plot for entropy
fig_entropy = px.bar(
    entropy_df,
    x="Entropy", y="n",
    color="Method",
    barmode="group",
    template="simple_white",
    color_discrete_map=dynamic_colors,  # Apply custom colors
)

# Add a custom message dynamically at the bottom-left corner
fig_entropy.add_annotation(
    text=missing_message,
    x=entropy_df['Entropy'].min() + 0.016,  # Use the minimum entropy value dynamically
    y=min(n_values) +2,  # Adjust to position below the first "n" value
    showarrow=False,
    font=dict(size=16, color="black"),
    align="left",
    bgcolor='rgba(255, 255, 255, 0.8)',
    bordercolor='rgba(202, 193, 193, 0.8)',
)

# Customize layout
fig_entropy.update_layout(
    height=900, width=1000,
    bargap=0.3,
    title=dict(
        text="Comparison of Change in Entropy Across DFT Methods_split-valence",
        font=dict(size=22, family="Arial, sans-serif", color="black"),
        x=0.5, y=0.97,  # Center the title
    ),
    xaxis=dict(
        title="Change in Entropy (kcal/mol·K)",
        titlefont=dict(size=18, family="Arial, sans-serif", color="black"),
        tickfont=dict(size=16, family="Arial, sans-serif"),
        range=(0,-0.13),
    ),
    yaxis=dict(
        title="Number of Carbon Atoms (n)",
        titlefont=dict(size=18, family="Arial, sans-serif", color="black"),
        tickfont=dict(size=16, family="Arial, sans-serif"),
    ),
    legend=dict(
        title="Method",
        title_font=dict(size=18, family="Arial, sans-serif"),
        font=dict(size=14, family="Arial, sans-serif"),
        x=0.7, y=0.05,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(202, 193, 193, 0.8)',
        borderwidth=1,
    ),
    plot_bgcolor='white',       # Background color for the plot area
    paper_bgcolor='rgba(248, 244, 244, 0.8)',      # Background color for the entire figure
    hovermode="x unified",      # Unified hover display
)

# Show the entropy bar plot
fig_entropy.show()

# Save as PNG
fig_entropy.write_image("bar_plot_split_entropy.png", format="png", scale=5)

# save a html
import plotly.io as pio
pio.write_html(fig_entropy, 'bar_plot_split_entropy.html', auto_open=False, )

# Save entropy data to the csv file after Gibbs free energy 
T = 298.15  # Room temperature

# Process CSV files and save entropy data
for file, method in zip(filenames, method_names):
    # Read the CSV file
    df = pd.read_csv(file)

    # Initialize a list for entropy values
    entropy_values = []

    # Calculate entropy for each row
    for _, row in df.iterrows():
        enthalpy = row['Reaction Enthalpy (kcal/mol)']
        gibbs = row['Gibbs Free Energy Change (kcal/mol)']

        if pd.notnull(enthalpy) and pd.notnull(gibbs):
            # Calculate entropy using ΔS = (ΔH - ΔG) / T
            delta_s = (enthalpy - gibbs) / T
            entropy_values.append(delta_s)
        else:
            entropy_values.append(None)  # Handle missing values

    # Add the entropy column to the DataFrame
    df['Entropy (kcal/mol·K)'] = entropy_values

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file, index=False)

print("Entropy data has been added and saved to the CSV files.")


