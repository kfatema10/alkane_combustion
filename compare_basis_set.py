import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go

# Load the data
data_file = 'enthalpy_data.csv'
df = pd.read_csv(data_file)

# Extract method names and group them by functional
methods = list(df.columns[:-1])  # Exclude experimental column
functionals = sorted(set(method.split('_')[0] for method in methods))
basis_sets = ['6-31G(d)', 'cc-pVDZ']

# Prepare data for Plotly
data = {functional: {basis: [] for basis in basis_sets} for functional in functionals}
for functional in functionals:
    for basis in basis_sets:
        method = f"{functional}_{basis}"
        if method in df.columns:
            data[functional][basis] = [df[method][i] if pd.notna(df[method][i]) else None for i in range(10)]

# Add experimental values
experimental_data = [df['enthalpy_expt'][i] for i in range(10)]

# Ensure subplot dimensions match the number of functionals
num_functionals = len(functionals)
rows = (num_functionals + 1) // 2
cols = 2

# Create subplot
fig = sp.make_subplots(
    rows=rows, 
    cols=cols,
    subplot_titles=functionals,
    vertical_spacing=0.1,
    shared_xaxes=True,  # Common X-axis
    shared_yaxes=True   # Common Y-axis
)

colors = {"6-31G(d)": "#afadab", "cc-pVDZ": "#fc9272", "Experimental": "#de2d26"}
n_values = [f"C{i+1}" for i in range(10)]

for i, functional in enumerate(functionals):
    row = (i // cols) + 1
    col = (i % cols) + 1

    # Add bars for each basis set
    for basis in basis_sets:
        fig.add_trace(
            go.Bar(
                x=n_values,
                y=data[functional][basis],
                name=basis if i == 0 else None,  # Show legend only once
                marker_color=colors[basis],
                showlegend=(i == 0)
            ),
            row=row,
            col=col
        )

    # Add experimental bars
    fig.add_trace(
        go.Bar(
            x=n_values,
            y=experimental_data,
            name="Experimental" if i == 0 else None,  # Show legend only once
            marker_color=colors["Experimental"],
            showlegend=(i == 0)
        ),
        row=row,
        col=col
    )

# Update layout
fig.update_layout(
    height=900,
    width=1000,
    title=dict(
        text="Comparison of Reaction Enthalpy across Basis Sets and Experimental Data",
        x=0.5,
        font=dict(size=20)
    ),
    barmode="group",
    legend=dict(
        title="Legend",
        font=dict(size=14),
        orientation="h",
        x=0.5,
        y=-0.15,  # Position legend below the plot
        xanchor="center",
        yanchor="top"
    ),
    plot_bgcolor="white"
)

# Update axes with common titles
font_style = dict(family="Arial", size=14, color="black")
fig.update_xaxes(
    title="Number of Carbon Atoms", 
    title_font=font_style, 
    tickfont=font_style, 
    tickvals=n_values, 
    ticktext=n_values,
)
fig.update_yaxes(
    title="Reaction Enthalpy (kcal/mol)", 
    title_font=font_style, 
    tickfont=font_style
)

# Add annotation for missing data
missing_annotation = "<b>Note:</b> Missing data for some n-values in PBEPBE and TPSSh with 6-31G(d)."
fig.add_annotation(
    text=missing_annotation,
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.12,
    showarrow=False,
    font=dict(size=14)
)

# Show the plot
# fig.show()
fig.show(renderer='chrome')


# Save as an image
fig.write_image("enthalpy_basis.png", format="png", scale=5)

# save html file for sharing interactive plot in GitHub
import plotly.io as pio
pio.write_html(fig, file="compare_basis.html", auto_open=True, )
