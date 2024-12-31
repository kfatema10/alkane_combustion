import numpy as np
import pandas as pd
from lmfit.models import LinearModel
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the experimental enthalpy data
enthalpy_expt = [-212.954, -373.088, -530.5927, -687.8585, -838.6711,
                 -994.9809, -1151.291, -1307.361, -1463.91, -1619.981]

# Define folder path and method to extract filenames
folder_path = Path('./')
filenames = sorted(folder_path.glob('*.csv'))

# Extract method names and data, last 2 words give method name as 'functional_basis set' 
method_names = []
for file in filenames:
    stem_parts = str(file.stem).split('_')  # Split the filename into parts
    method_name = f"{stem_parts[-2]}_{stem_parts[-1]}"  # Combine the last two parts
    method_names.append(method_name)

# Initialize lists for plotting
n_values = np.array(range(1, 11))  # n=1 to n=10
enthalpy_data = {method: [None] * len(n_values) for method in method_names}

# Process CSV files
for file, method in zip(filenames, method_names):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        n = int(row['No. of C atom'])
        enthalpy_data[method][n - 1] = row['Reaction Enthalpy (kcal/mol)']

# Fit enthalpy data and create a subplot
fig1 = make_subplots(rows=2, cols=3, subplot_titles=method_names, shared_xaxes=True, shared_yaxes=True)

colors = ["red", "blue", "green", "orange", "purple", "brown"]  # Colors for plots
font_style = dict(family="Arial", size=12, color="black")

for i, (method, enthalpies) in enumerate(enthalpy_data.items()):
    # Remove None values and corresponding n values
    valid_indices = [j for j, value in enumerate(enthalpies) if value is not None]
    x_valid = n_values[valid_indices]
    y_valid = np.array(enthalpies)[valid_indices]

    # Fit a linear model
    model = LinearModel()
    params = model.make_params(slope=1, intercept=0)
    result = model.fit(y_valid, params, x=x_valid)

    # Generate fitted data
    x_fit = np.linspace(min(x_valid), max(x_valid), 100)
    y_fit = result.eval(x=x_fit)

    # Calculate R-squared
    r_squared = result.rsquared

    # Add subplot
    row = i // 3 + 1
    col = i % 3 + 1
     
    
    # Add experimental data as scatter plot (same for all subplots)
    fig1.add_trace(go.Scatter(x=n_values, y=enthalpy_expt, mode="markers", name="Experimental Data",
                             marker=dict(symbol='star', color='rgba(90, 85, 85, 0.8)', size=8, line=dict(width=1, color='white')),
                            #  showlegend=(i == 0)),  # Show legend only for the first subplot
                            showlegend=False),
                  row=row, col=col)

    fig1.add_trace(go.Scatter(x=x_valid, y=y_valid, mode="markers", name=f"{method} Data",
                             marker=dict(color=colors[i])),  
                  row=row, col=col)  

    fig1.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name=f"{method} Fit: y = {result.params['slope'].value:.2f}x + {result.params['intercept'].value:.2f}, R² = {r_squared:.2f}",
                             line=dict(color=colors[i])),
                  row=row, col=col)
    
# Update layout
fig1.update_layout(
    width=1000, height=700,
    title=dict(
                text="Linear Fit of Reaction Enthalpies_split-valence",
                font=dict(size=22, family="Arial", color="black"),
                x=0.5, y=0.97,  # Center the title
                ),
    font=font_style,
    showlegend=True,
    legend=dict(x=0, y=-0.23, font=font_style, orientation="h"),
    plot_bgcolor='white',  # Background color for the plot area
    paper_bgcolor='rgba(248, 244, 244, 0.8)',  # Background color for the entire figure
)

# Add annotation for the experimental data legend
fig1.add_annotation(
    x=0.03,  # left side below the plot
    y=-0.2,  # Position below the legend
    text="Gray star indicates the Experimental value",
    showarrow=False,
    font=dict(size=14, color="black", family="Arial"),
    xref="paper",
    yref="paper",
    align="center",
    bgcolor="rgba(255, 255, 255, 0.8)",  # Light background to enhance visibility
    bordercolor="rgba(226, 214, 214, 0.8)",
    borderwidth=1,
)

# Update axes
fig1.update_xaxes(title="Number of C Atoms", title_font=font_style, tickfont=font_style, tickvals=n_values, ticktext=[f"{n}" for n in n_values])
fig1.update_yaxes(title="Reaction Enthalpy (kcal/mol)", title_font=font_style, tickfont=font_style)

# Show the interactive plot
fig1.show()

# Save as PNG
fig1.write_image("fit_split_enthalpy.png", format="png", scale=5)

# save a html
import plotly.io as pio
pio.write_html(fig1, 'fit_split_enthalpy.html', auto_open=False, )

# ----------------------------
# Initialize lists for plotting fitted eqn for gibbs
n_values = np.array(range(1, 11))  # n=1 to n=10
gibbs_data = {method: [None] * len(n_values) for method in method_names}

# Process CSV files
for file, method in zip(filenames, method_names):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        n = int(row['No. of C atom'])
        gibbs_data[method][n - 1] = row['Gibbs Free Energy Change (kcal/mol)']

# Fit enthalpy data and create a subplot
fig2 = make_subplots(rows=2, cols=3, subplot_titles=method_names, shared_xaxes=True, shared_yaxes=True)

colors = ["red", "blue", "green", "orange", "purple", "brown"]  # Colors for plots
font_style = dict(family="Arial", size=12, color="black")

for i, (method, gibbs) in enumerate(gibbs_data.items()):
    # Remove None values and corresponding n values
    valid_indices = [j for j, value in enumerate(gibbs) if value is not None]
    x_valid = n_values[valid_indices]
    y_valid = np.array(gibbs)[valid_indices]

    # Fit a linear model
    model = LinearModel()
    params = model.make_params(slope=1, intercept=0)
    result = model.fit(y_valid, params, x=x_valid)

    # Generate fitted data
    x_fit = np.linspace(min(x_valid), max(x_valid), 100)
    y_fit = result.eval(x=x_fit)

    # Calculate R-squared
    r_squared = result.rsquared

    # Add subplot
    row = i // 3 + 1
    col = i % 3 + 1
    
    fig2.add_trace(go.Scatter(x=x_valid, y=y_valid, mode="markers", name=f"{method} Data",
                             marker=dict(color=colors[i])),  
                  row=row, col=col)   
    fig2.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", 
                                name=f"{method} Fit: y = {result.params['slope'].value:.2f}x + {
                                    result.params['intercept'].value:.2f}, R² = {r_squared:.2f}",
                             line=dict(color=colors[i])),
                  row=row, col=col)

# Update layout
fig2.update_layout(
    width=1000, height=700,
    title=dict(
                text="Linear Fit of Change in Gibbs Energy_split-valence",
                font=dict(size=22, family="Arial", color="black"),
                x=0.5, y=0.97,  # Center the title
                ),
    font=font_style,
    showlegend=True,
    legend=dict(x=0, y=-0.3, font=font_style, orientation="h"),
    plot_bgcolor='white',  # Background color for the plot area
    paper_bgcolor='rgba(248, 244, 244, 0.8)',  # Background color for the entire figure
)
 
# Update axes
fig2.update_xaxes(title="Number of C Atoms", title_font=font_style, tickfont=font_style, tickvals=n_values, ticktext=[f"{n}" for n in n_values])
fig2.update_yaxes(title="Chnage in Gibbs Energy (kcal/mol)", title_font=font_style, tickfont=font_style)

# Show the interactive plot
fig2.show()

# Save as PNG
fig2.write_image("fit_split_gibbs.png", format="png", scale=5)

# save a html
import plotly.io as pio
pio.write_html(fig2, 'fit_split_gibbs.html', auto_open=False, )

# ----------------------------
# Initialize lists for plotting fitted eqn for entropy
n_values = np.array(range(1, 11))  # n=1 to n=10
entropy_data = {method: [None] * len(n_values) for method in method_names}

# Process CSV files
for file, method in zip(filenames, method_names):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        n = int(row['No. of C atom'])
        entropy_data[method][n - 1] = row['Entropy (kcal/mol·K)']

# Fit enthalpy data and create a subplot
fig3 = make_subplots(rows=2, cols=3, subplot_titles=method_names, shared_xaxes=True, shared_yaxes=True)

colors = ["red", "blue", "green", "orange", "purple", "brown"]  # Colors for plots
font_style = dict(family="Arial", size=12, color="black")

for i, (method, entropies) in enumerate(entropy_data.items()):
    # Remove None values and corresponding n values
    valid_indices = [j for j, value in enumerate(entropies) if value is not None]
    x_valid = n_values[valid_indices]
    y_valid = np.array(entropies)[valid_indices]

    # Fit a linear model
    model = LinearModel()
    params = model.make_params(slope=1, intercept=0)
    result = model.fit(y_valid, params, x=x_valid)

    # Generate fitted data
    x_fit = np.linspace(min(x_valid), max(x_valid), 100)
    y_fit = result.eval(x=x_fit)

    # Calculate R-squared
    r_squared = result.rsquared

    # Add subplot
    row = i // 3 + 1
    col = i % 3 + 1
    fig3.add_trace(go.Scatter(x=x_valid, y=y_valid, mode="markers", name=f"{method} Data",
                             marker=dict(color=colors[i])),  
                  row=row, col=col)   
    fig3.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", 
                            name=f"{method} Fit: y = {result.params['slope'].value:.4f}x + {result.params['intercept'].value:.4f}, R² = {r_squared:.2f}",
                            line=dict(color=colors[i])),
                  row=row, col=col)

# Update layout
fig3.update_layout(
    width=1000, height=700,
    title=dict(
                text="Linear Fit of Reaction Entropies_split-valence",
                font=dict(size=22, family="Arial", color="black"),
                x=0.5, y=0.97,  # Center the title
                ),
    # title="Linear Fit of Reaction Entropies",
    # title_font=dict(family="Arial", size=18, color="black",),
    font=font_style,
    showlegend=True,
    legend=dict(x=0, y=-0.3, font=font_style, orientation="h"),
    plot_bgcolor='white',  # Background color for the plot area
    paper_bgcolor='rgba(248, 244, 244, 0.8)',  # Background color for the entire figure
)
 
# Update axes
fig3.update_xaxes(title="Number of C Atoms", title_font=font_style, tickfont=font_style, tickvals=n_values, ticktext=[f"{n}" for n in n_values])
fig3.update_yaxes(title="Reaction Entropy (kcal/mol.K)", title_font=font_style, tickfont=font_style)

# Show the interactive plot
fig3.show()

# Save as PNG
fig3.write_image("fit_split_entropy.png", format="png", scale=5)

# save a html
import plotly.io as pio
pio.write_html(fig3, 'fit_split_entropy.html', auto_open=False, )
