import streamlit as st
import pandas as pd
import numpy as np
from pyvis.network import Network
import tempfile
import os
from scipy.stats import chi2_contingency
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Association Explorer", layout="wide")

st.title("Association Explorer")

# Helper functions for association measures
def cramers_v(x, y):
    """Compute Cramer's V for two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def correlation_ratio(categories, measurements):
    """Compute eta (correlation ratio) for categorical vs numeric."""
    categories = pd.Series(categories)
    measurements = pd.Series(measurements)
    # Remove missing values
    mask = categories.notna() & measurements.notna()
    categories = categories[mask]
    measurements = measurements[mask]
    
    if len(measurements) == 0:
        return 0.0
    
    # Overall mean
    mean_global = measurements.mean()
    # Group means
    groups = measurements.groupby(categories)
    # Between-group variance
    ssb = sum(len(group) * (group.mean() - mean_global) ** 2 for _, group in groups)
    # Total variance
    sst = sum((measurements - mean_global) ** 2)
    
    if sst == 0:
        return 0.0
    
    eta_squared = ssb / sst
    return np.sqrt(max(0, eta_squared))


# Load the CSV file
df = pd.read_csv("data/data.csv")

# Optional variable descriptions
descriptions = {}
try:
    desc_df = pd.read_csv("data/description.csv")
    if {"Variable", "Description"}.issubset(desc_df.columns):
        descriptions = desc_df.set_index("Variable")["Description"].to_dict()
except FileNotFoundError:
    descriptions = {}

# Keep both numeric and categorical variables, exclude constants
selected_cols = []
for col in df.columns:
    if df[col].nunique(dropna=True) > 1:
        selected_cols.append(col)

df_filtered = df[selected_cols].copy()

# Identify numeric vs categorical columns
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df_filtered.columns if c not in numeric_cols]

threshold = st.slider(
    "Association threshold", 
    0.0, 1.0, (0.1, 1.0), 0.01,
    help="Threshold applies to r² (numeric–numeric), η² (numeric–categorical), and Cramér's V (categorical–categorical)."
)
threshold_min, threshold_max = threshold

# Compute all pairwise associations
edges = []
nodes_in_use = set()

all_cols = df_filtered.columns.tolist()
for i in range(len(all_cols)):
    for j in range(i + 1, len(all_cols)):
        col_i, col_j = all_cols[i], all_cols[j]
        
        # Determine variable types
        is_i_numeric = col_i in numeric_cols
        is_j_numeric = col_j in numeric_cols
        
        # Drop missing values for this pair
        pair_data = df_filtered[[col_i, col_j]].dropna()
        if len(pair_data) < 2:
            continue
        
        if is_i_numeric and is_j_numeric:
            # Numeric - Numeric: Pearson correlation
            corr_val = pair_data[col_i].corr(pair_data[col_j])
            if pd.isna(corr_val):
                continue
            strength = corr_val ** 2  # r² for thresholding
            assoc_type = "Pearson r"
            assoc_value = corr_val
            threshold_value = strength  # r²
            edge_color = "#d62728" if corr_val < 0 else "#4f81bd"
            
        elif not is_i_numeric and not is_j_numeric:
            # Categorical - Categorical: Cramer's V
            v = cramers_v(pair_data[col_i], pair_data[col_j])
            strength = v  # V for thresholding
            assoc_type = "Cramér's V"
            assoc_value = v
            threshold_value = strength  # V
            edge_color = "#7f7f7f"  # gray
            
        else:
            # Numeric - Categorical: Correlation ratio (eta)
            if is_i_numeric:
                eta = correlation_ratio(pair_data[col_j], pair_data[col_i])
            else:
                eta = correlation_ratio(pair_data[col_i], pair_data[col_j])
            strength = eta ** 2  # η² for thresholding
            assoc_type = "Eta (η)"
            assoc_value = eta
            threshold_value = strength  # η²
            edge_color = "#ff7f0e"  # orange
        
        # Filter by threshold
        if threshold_min <= threshold_value <= threshold_max:
            edges.append((col_i, col_j, strength, assoc_type, assoc_value, threshold_value, edge_color))
            nodes_in_use.update([col_i, col_j])

if not nodes_in_use:
    st.info("No associations within the selected threshold range.")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Correlation network", "Pair plots", "Help"])

# ============================================================================
# TAB 1: CORRELATION NETWORK
# ============================================================================
with tab1:
    net = Network(height="700px", width="100%", bgcolor="white", font_color="black")
    net.barnes_hut()

    for col in sorted(nodes_in_use):
        desc_text = descriptions.get(col, col)
        net.add_node(col, label=col, title=f"{col}: {desc_text}", font={"size": 20})

    for src, dst, strength, assoc_type, assoc_value, threshold_value, edge_color in edges:
        if assoc_type == "Pearson r":
            tooltip = f"{src} ↔ {dst}\nPearson r: {assoc_value:.3f}\nThreshold metric (r²): {threshold_value:.3f}"
        elif assoc_type == "Eta (η)":
            tooltip = f"{src} ↔ {dst}\nEta (η): {assoc_value:.3f}\nThreshold metric (η²): {threshold_value:.3f}"
        elif assoc_type == "Cramér's V":
            tooltip = f"{src} ↔ {dst}\nCramér's V: {assoc_value:.3f}"
        else:
            tooltip = f"{src} ↔ {dst}\n{assoc_type}: {assoc_value:.3f}"
        
        net.add_edge(
            src,
            dst,
            value=float(strength * 5),  # scale for visibility
            title=tooltip,
            color=edge_color
        )

    net.set_options(
            """
            {
                "nodes": {
                    "shape": "dot",
                    "size": 20,
                    "borderWidth": 1,
                    "font": { "size": 20, "color": "#1f2933" },
                    "scaling": { "min": 20, "max": 28, "label": { "enabled": true, "min": 20, "max": 28 } }
                },
                "edges": {
                    "color": { "color": "#4f81bd", "highlight": "#2b3a67" },
                    "width": 2,
                    "smooth": true
                },
                "physics": {
                    "stabilization": true,
                    "barnesHut": {
                        "gravitationalConstant": -12000,
                        "springLength": 220,
                        "springConstant": 0.02
                    }
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 120,
                    "zoomView": true,
                    "zoomSpeed": 0.6
                }
            }
            """
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, "r", encoding="utf-8").read()

    st.components.v1.html(html, height=750, scrolling=True)


# ============================================================================
# TAB 2: PAIR PLOTS
# ============================================================================
with tab2:
    
    # Filter input
    search_query = st.text_input("Filter by variable name", "").lower()
    
    # Filter edges based on search query
    filtered_edges = [
        e for e in edges 
        if (search_query in e[0].lower() or search_query in e[1].lower())
    ]
    
    if not filtered_edges:
        st.info("No pairs match your search.")
    else:
        for idx, (src, dst, strength, assoc_type, assoc_value, threshold_value, edge_color) in enumerate(filtered_edges):
            # Determine variable types
            is_src_numeric = src in numeric_cols
            is_dst_numeric = dst in numeric_cols
            
            with st.expander(f"{idx + 1}. {src} ↔ {dst}"):
                # Get pair data
                pair_data = df_filtered[[src, dst]].dropna()
                
                # Helper to get description or fallback to variable name
                def get_label(var):
                    return descriptions.get(var, var)
                
                if is_src_numeric and is_dst_numeric:
                    # ===== NUMERIC - NUMERIC: SCATTERPLOT WITH REGRESSION LINE =====
                    src_label = get_label(src)
                    dst_label = get_label(dst)
                    
                    # Create dataframe with jittered values for display but keep originals for tooltip
                    plot_data = pair_data[[src, dst]].copy()
                    plot_data['original_x'] = plot_data[src]
                    plot_data['original_y'] = plot_data[dst]
                    
                    # Add jitter to avoid point overlap
                    jitter_amount = 0.02
                    x_range = plot_data[src].max() - plot_data[src].min()
                    y_range = plot_data[dst].max() - plot_data[dst].min()
                    plot_data[src] = plot_data[src] + np.random.uniform(-jitter_amount * x_range, jitter_amount * x_range, len(plot_data))
                    plot_data[dst] = plot_data[dst] + np.random.uniform(-jitter_amount * y_range, jitter_amount * y_range, len(plot_data))
                    
                    # Create scatterplot with trendline
                    fig = px.scatter(
                        plot_data,
                        x=src,
                        y=dst,
                        custom_data=['original_x', 'original_y'],
                        trendline="ols",
                        trendline_color_override="#d62728" if assoc_value < 0 else "#4f81bd"
                    )
                    
                    # Update axis labels with descriptions
                    fig.update_xaxes(title_text=src_label)
                    fig.update_yaxes(title_text=dst_label)
                    
                    # Update hover for scatter points to show original (non-jittered) values
                    fig.update_traces(
                        hovertemplate=f"<b>{src}</b>: %{{customdata[0]:.2f}}<br><b>{dst}</b>: %{{customdata[1]:.2f}}<extra></extra>",
                        selector=dict(mode="markers")
                    )
                    
                    # Update hover for trendline to show only R² formatted to 3 digits
                    fig.update_traces(
                        hovertemplate=f"<b>R²:</b> {threshold_value:.3f}<extra></extra>",
                        selector=dict(mode="lines")
                    )
                    
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        title="",
                        hovermode="closest"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif not is_src_numeric and not is_dst_numeric:
                    # ===== CATEGORICAL - CATEGORICAL: CONTINGENCY TABLE HEATMAP =====
                    contingency = pd.crosstab(pair_data[src], pair_data[dst], margins=True)
                    contingency.index.name = None
                    contingency.columns.name = None
                    src_label = get_label(src)
                    dst_label = get_label(dst)
                    st.write(f"**{src_label}** × **{dst_label}**")
                    st.dataframe(contingency, use_container_width=True)
                    
                    # Heatmap without margins for visualization
                    cont_no_margins = pd.crosstab(pair_data[src], pair_data[dst])
                    src_label = get_label(src)
                    dst_label = get_label(dst)
                    
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=cont_no_margins.values,
                            x=cont_no_margins.columns,
                            y=cont_no_margins.index,
                            colorscale="Blues",
                            text=cont_no_margins.values,
                            texttemplate="%{text}",
                            textfont={"size": 10},
                            hoverinfo="skip",
                            showscale=False
                        )
                    )
                    fig.update_layout(
                        xaxis_title=dst_label,
                        yaxis_title=src_label,
                        height=400,
                        title=""
                    )
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # ===== NUMERIC - CATEGORICAL: MEAN PLOT =====
                    # Identify which is numeric and which is categorical
                    if is_src_numeric:
                        numeric_var = src
                        cat_var = dst
                    else:
                        numeric_var = dst
                        cat_var = src
                    
                    # Compute means by category, sorted in descending order
                    means = pair_data.groupby(cat_var)[numeric_var].mean().sort_values(ascending=False)
                    
                    numeric_label = get_label(numeric_var)
                    cat_label = get_label(cat_var)
                    
                    # Create bar plot
                    fig = px.bar(
                        x=means.index,
                        y=means.values,
                        labels={"x": cat_label, "y": f"Mean of '{numeric_label}'"}
                    )
                    
                    # Add mean values inside bars
                    fig.update_traces(
                        text=[f"{v:.2f}" for v in means.values],
                        textposition="inside",
                        hovertemplate=None,
                        hoverinfo="skip"
                    )
                    fig.update_layout(height=400, showlegend=False, title="")
                    st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3: HELP =====
with tab3:
    st.header("About this Application")
    
    st.markdown("""
    This application helps you explore associations between variables in your dataset using multiple statistical measures.
    
    ### Association Measures
    
    **Numeric ↔ Numeric Variables:**
    - **Pearson correlation coefficient (r)**: Measures linear relationship between two continuous variables
    - Ranges from -1 (perfect negative) to +1 (perfect positive correlation)
    - **Threshold applied on r²**: The proportion of variance explained
    
    **Categorical ↔ Categorical Variables:**
    - **Cramér's V**: Measures association between two categorical variables
    - Ranges from 0 (no association) to 1 (perfect association)
    - **Threshold applied on V directly**: The strength of association
    
    **Numeric ↔ Categorical Variables:**
    - **Correlation ratio (η)**: Measures how well a categorical variable explains variance in a numeric variable
    - Ranges from 0 (no relationship) to 1 (perfect relationship)
    - **Threshold applied on η²**: The proportion of variance explained by group membership
    
    ### How to Use
    
    1. **Set the association threshold** using the range slider to filter relationships by strength
    2. **Correlation Network tab**: View an interactive network graph showing all associations above the threshold
       - Node size represents variables
       - Edge colors: Blue (positive correlation), Red (negative correlation), Orange (eta), Gray (Cramér's V)
       - Hover over nodes and edges for detailed information
    3. **Pair Plots tab**: Explore individual variable relationships in detail
       - Filter pairs by variable name
       - View scatter plots with regression lines for numeric pairs
       - View contingency tables and heatmaps for categorical pairs
       - View mean plots for numeric-categorical pairs
    
    ### Tips
    
    - Lower threshold values (e.g., 0.1-0.3) show more associations including weaker ones
    - Higher threshold values (e.g., 0.5+) focus on stronger relationships
    - Use the search box in Pair Plots to quickly find specific variable combinations
    - Variable descriptions (if provided) appear in tooltips and axis labels
    """)
    
    st.markdown("---")
    st.markdown("*AssociationExplorer. [Code](https://github.com/AntoineSoetewey/AssociationExplorer-python)*")