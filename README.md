# AssociationExplorer - Python

A **Streamlit** application to explore **associations between variables** in a tabular dataset.  
It combines an **interactive association network** (PyVis) with **pairwise visualizations** (Plotly) for all variable pairs that fall within a user-defined association threshold.

## Features

### 1) Association Network
- Builds a network where:
  - **nodes** are variables
  - **edges** are associations that pass the threshold
- Hover tooltips show the association measure and the threshold metric used.
- Edge color encodes association type:
  - **Blue / Red**: numeric–numeric (Pearson *r*, sign indicates direction)
  - **Orange**: numeric–categorical (Eta *η*)
  - **Gray**: categorical–categorical (Cramér’s *V*)
- Interpreting the Network Visualization:
  - **Edge thickness** is proportional to the **association strength**; thicker edges indicate stronger associations
  - **Distance between nodes** is determined by a force-directed layout algorithm. Variables connected by stronger associations tend to appear closer, but distances should be interpreted **qualitatively**, not quantitatively

### 2) Pair Plots
For every association kept in the network, the app displays the corresponding “pair plot”:
- **numeric ↔ numeric**: scatterplot + regression line (OLS)
- **categorical ↔ categorical**: contingency table + heatmap
- **numeric ↔ categorical**: mean plot (bar chart with mean shown on each bar)

Includes a search box to filter by variable name.

### 3) Help tab
A short, user-friendly explanation of the association measures and how to interpret the threshold.

## Association measures and thresholding

The app computes pairwise associations depending on variable types and applies the threshold to:

- **numeric ↔ numeric**: Pearson correlation *r*  
  - threshold metric: **r²**
- **numeric ↔ categorical**: correlation ratio *η*  
  - threshold metric: **η²**
- **categorical ↔ categorical**: Cramér’s *V*  
  - threshold metric: **V**

> The threshold slider is a **range**: only associations with  
> `threshold_min <= metric_used_for_threshold <= threshold_max`  
> are displayed.

## Quick start

```bash
# 1) Create and activate a virtual env (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the Streamlit app
streamlit run app.py
```

If Streamlit complains about `ModuleNotFoundError: No module named 'pyvis'`, it means dependencies were not installed in the active environment. Activate the virtual environment (step 1) and rerun the install command (step 2).

## License

MIT License

## Inspiration

This application is inspired by the following R Shiny projects:

- **AssociationExplorer**  
  https://github.com/AntoineSoetewey/AssociationExplorer

- **AssociationExplorer2**  
  https://github.com/AntoineSoetewey/AssociationExplorer2

The goal of this Python version is not to replicate these applications feature-by-feature, but to provide a **Streamlit-based implementation** that follows the same statistical principles for exploratory association analysis.
