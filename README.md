# GraphSD

**GraphSD** (Graph-based Subgroup Discovery) is a Python package for detecting exceptional interaction patterns in graphs. It builds spatio-temporal graphs from position and attribute data, then applies rule-based subgroup discovery and outlier detection techniques to uncover meaningful and rare behaviors.

[![PyPI version](https://badge.fury.io/py/graph-sd.svg)](https://pypi.org/project/graph-sd/)  
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

## ✨ Features

- Directed and multi-directed interaction graph construction  
- Subgroup discovery using interpretable rule-based conditions  
- Outlier detection and quality-based ranking  
- Spatio-temporal interaction filtering using distance and velocity  
- Binning and discretization utilities  
- Built-in graph visualizations with pattern overlays  
- Pure Python: no dependency on Orange3 or external mining engines

---

## 📦 Installation

Install via PyPI:

```bash
pip install graph-sd
```

---

## 🚀 Example Usage

```python
from graphsd.mining import DigraphSDMining
from graphsd.utils import make_bins
from graphsd._base import load_data
from graphsd.viz import graph_viz
import networkx as nx

# Load sample position and social data
position_df, social_df = load_data("playground_a")

# Discretize social attributes
social_df = make_bins(social_df)

# Initialize the subgroup discovery engine
dig = DigraphSDMining(random_state=42)

# Build the interaction graph using position and attribute data
dig.read_data(position_df, social_df, time_step=10)

# Discover subgroups with quality constraints
subgroups = dig.subgroup_discovery(
    mode="to",
    min_support=0.2,
    metric="mean",
    quality_measure="global_proportion"
)

# Convert to a DataFrame and print
df = dig.to_dataframe(subgroups)
print(df)

# Visualize the graph and highlighted subgroups
graph_viz(dig.graph, layout=nx.spring_layout)
```

---

## 🧠 Code Structure

| Module        | Purpose |
|---------------|---------|
| `mining.py`   | Main API for graph construction and subgroup discovery |
| `patterns.py` | Logic for rule quality, coverage, and pattern filters |
| `outlier.py`  | Tools for subgroup scoring and ranking |
| `utils.py`    | Preprocessing, binning, and distance computations |
| `viz.py`      | Graph and subgroup visualizations |
| `_base.py`    | Sample data loader (e.g. `load_data("playground_a")`) |

---

## 📄 License

This project is licensed under the **BSD 3-Clause License**.

---

## 👥 Authors

- **Carolina Centeio Jorge** – TU Delft  
- **Cláudio Rebelo de Sá** – Leiden University

---

## 🌐 Links

- 📦 [PyPI Package](https://pypi.org/project/graph-sd/)  
- 🧑‍💻 [GitHub Repository](https://github.com/centeio/GraphSD)

---

## 📚 Citation

If you use **GraphSD** in your research, please cite:

### 📝 Journal Article (Expert Systems, 2023)

> Jorge, C.C., Atzmueller, M., Heravi, B.M., Gibson, J.L., Rossetti, R.J.F., & Rebelo de Sá, C.  
> *"Want to come play with me?" Outlier subgroup discovery on spatio-temporal interactions*.  
> Expert Systems, 40(5), 2023.  
> [https://doi.org/10.1111/exsy.12686](https://doi.org/10.1111/exsy.12686)

```bibtex
@article{DBLP:journals/es/JorgeAHGRS23,
  author  = {Carolina Centeio Jorge and Martin Atzmueller and Behzad Momahed Heravi and
             Jenny L. Gibson and Rosaldo J. F. Rossetti and Cl{'a}udio Rebelo de S{'a}},
  title   = {"Want to come play with me?" Outlier subgroup discovery on spatio-temporal interactions},
  journal = {Expert Syst. J. Knowl. Eng.},
  volume  = {40},
  number  = {5},
  year    = {2023},
  doi     = {10.1111/EXSY.12686}
}
```

### 📘 Conference Paper (EPIA 2019)

> Jorge, C.C., Atzmueller, M., Heravi, B.M., Gibson, J.L., Rebelo de Sá, C., & Rossetti, R.J.F.  
> *Mining Exceptional Social Behaviour*. In *EPIA 2019*, LNCS 11805, Springer.  
> [https://doi.org/10.1007/978-3-030-30244-3_38](https://doi.org/10.1007/978-3-030-30244-3_38)

```bibtex
@inproceedings{DBLP:conf/epia/JorgeAHGSR19,
  author    = {Carolina Centeio Jorge and Martin Atzmueller and Behzad Momahed Heravi and
               Jenny L. Gibson and Cl{'a}udio Rebelo de S{'a} and Rosaldo J. F. Rossetti},
  title     = {Mining Exceptional Social Behaviour},
  booktitle = {Progress in Artificial Intelligence - 19th EPIA 2019},
  series    = {Lecture Notes in Computer Science},
  volume    = {11805},
  pages     = {460--472},
  publisher = {Springer},
  year      = {2019},
  doi       = {10.1007/978-3-030-30244-3_38}
}
```
