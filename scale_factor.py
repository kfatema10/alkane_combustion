import pandas as pd
import plotly.graph_objects as go

# Load the data
data_file = 'enthalpy_data.csv'
df = pd.read_csv(data_file)

# Extract method names and experimental data
methods = list(df.columns[:-1])  # Exclude experimental column
experimental_data = df['enthalpy_expt'].values

# Calculate scaling factors
scale_factors = pd.DataFrame(index=df.index, columns=methods)

for method in methods:
    dft_data = df[method].values
    for i in range(len(dft_data)):
        if pd.notna(dft_data[i]):
            scale_factors.loc[i, method] = experimental_data[i] / dft_data[i]
        else:
            # Average the previous and next scaling factors for empty cells
            prev_value = scale_factors.loc[i - 1, method] if i > 0 else None
            next_value = scale_factors.loc[i + 1, method] if i < len(dft_data) - 1 else None
            if pd.notna(prev_value) and pd.notna(next_value):
                scale_factors.loc[i, method] = (prev_value + next_value) / 2

# Save scaling factors to a CSV file
output_file = 'scaling.csv'
scale_factors.to_csv(output_file, index=False)

n_values = [f"C{i+1}" for i in range(10)]

# Plot the scaling factors using Plotly
fig = go.Figure()

# Add scatter plots for each method 

colors = ["#cb0fa3", "#0bb4ff", "#2cb164", "#676209", "#d80609", "#ffa300", 
            "#f012c8", "#296dc7", "#08d2b0",  "#a765ea", "#4421af", "#93e117",]

color_options = {method: colors[i % len(colors)] for i, method in enumerate(methods)}

for method in methods:
    fig.add_trace(go.Scatter(
        x=n_values,  # Number of C atoms (1 to 10),  x=df.index + 1, 
        y=scale_factors[method],
        mode='markers',
        marker=dict(color=color_options[method], size=10,),
        name=method,
    ))
 
# Update layout
fig.update_layout(
    height=900,
    width=1000,
    title=dict(
        text="Scaling Factors for Reacton Enthalpy",
        x=0.5,
        y=0.93,
        font=dict(size=20)
    ),
    barmode="group",
    legend=dict(
        title="Methods",
        font=dict(size=14),
        orientation="h",
        x=0.5,
        y=-0.15,  # Position legend below the plot
        xanchor="center",
        yanchor="top"
    ),
    plot_bgcolor="#e6e6e1"
)

# Update axes with common titles
font_style = dict(family="Arial", size=18, color="blue")

fig.update_xaxes(
    title="Number of Carbon Atoms", 
    title_font=font_style, 
    tickfont=dict(family="Arial", size=14, color="black"), 
    tickvals=n_values, 
    ticktext=n_values,
    
)
fig.update_yaxes(
    title="Scaling Factor", 
    title_font=font_style, 
    tickfont= dict(family="Arial", size=14, color="black"), 
)

# Add annotation for missing data
missing_annotation = "<b>Note:</b> Missing data for some n-values in PBEPBE and TPSSh with 6-31G(d)."
fig.add_annotation(
    text=missing_annotation,
    xref="paper",
    yref="paper",
    x=0.0,
    y=-0.12,
    showarrow=False,
    font=dict(size=16)
)


# Show plot
fig.show()

# Save as an image
fig.write_image("scale_factor.png", format="png", scale=5)

# save html file for sharing interactive plot in GitHub
import plotly.io as pio
pio.write_html(fig, file="scale_factor.html", auto_open=True, )
