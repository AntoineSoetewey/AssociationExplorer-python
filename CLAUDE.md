# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

There are no tests or linting configured for this project.

## Architecture

The entire application lives in a single file: [app.py](app.py). There is no module structure.

**Data flow:**

1. `data/data.csv` is loaded at startup — this is the dataset users explore. A optional `data/description.csv` (columns: `Variable`, `Description`) provides human-readable labels for axis titles and tooltips.
2. Constant columns (`.nunique() <= 1`) are dropped. Remaining columns are split into `numeric_cols` and `categorical_cols`.
3. A global threshold slider controls which pairs are displayed. All pairwise associations are computed upfront and filtered against `[threshold_min, threshold_max]`.
4. Passing pairs are stored as `edges` — a list of tuples `(src, dst, strength, assoc_type, assoc_value, threshold_value, edge_color)`.

**Three tabs rendered from `edges`:**

- **Tab 1 – Correlation network**: PyVis `Network` object built from `edges`, saved to a temp `.html` file, and embedded via `st.components.v1.html`.
- **Tab 2 – Pair plots**: Iterates `edges`, renders Plotly figures per pair type (scatter+OLS, heatmap, bar chart). A text filter narrows visible pairs.
- **Tab 3 – Help**: Static markdown.

**Association measures and thresholding:**

| Pair type | Measure | Threshold metric |
|---|---|---|
| numeric ↔ numeric | Pearson r | r² |
| numeric ↔ categorical | Correlation ratio η | η² |
| categorical ↔ categorical | Cramér's V | V |

Edge colors: blue (`#4f81bd`) for positive Pearson r, red (`#d62728`) for negative, orange (`#ff7f0e`) for η, gray (`#7f7f7f`) for Cramér's V. Edge `value` in PyVis is `strength * 5` (controls thickness).
