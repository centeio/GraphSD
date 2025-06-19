# GraphSD

**GraphSD** (Graph-based Subgroup Discovery) is a Python package for mining unusual interaction patterns in graphs using subgroup discovery techniques. It is designed for exploratory graph analysis in domains such as social networks, communication logs, and collaboration graphs.

[![PyPI version](https://badge.fury.io/py/graph-sd.svg)](https://pypi.org/project/graph-sd/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

## ✨ Features

- Subgroup Discovery over graph structures
- Pattern mining using Orange3-Associate
- Integration with popular Python libraries (`networkx`, `pandas`, `scipy`)
- Outlier detection
- Built-in visualizations

---

## 📦 Installation

```bash
pip install graph-sd
```
---

## 🚀 Getting Started

### Example Using Playground A Dataset

```python
from graphsd.datasets import load_playground_a, load_playground_b
from graphsd.mining import DigraphSDMining, MultiDigraphSDMining
from graphsd.utils import make_bins
from graphsd.viz import graph_viz
import networkx as nx


# Load datasets
position_data, social_data = load_playground_a()
# For Playground B instead, uncomment the next line:
# position_data, social_data = load_playground_b()

# Create bins for social data
social_data = make_bins(social_data)

# Choose a random seed for reproducibility
dig = DigraphSDMining(random_state=1234)

# Assign attributes to edges based on datasets
counter = dig.read_data(position_data, social_data, time_step=10)

# Run subgroup discovery on the digraph.
subgroups_to = dig.subgroup_discovery(
    mode="to",
    min_support=0.20,
    metric='mean',
    quality_measure="qP"
)

# Convert results to DataFrame
result_df = dig.to_dataframe(subgroups_to)
print(result_df.head())

# Visualize the graph with discovered subgroups
graph_viz(dig.graph, width=2, layout=nx.spring_layout)
```

---

## 📄 License

BSD 3-Clause License

---

## 👥 Authors

- C. Centeio Jorge – TU Delft
- Cláudio Rebelo de Sá – Leiden University

---

## 🌐 Links

- 📦 [PyPI Package](https://pypi.org/project/graph-sd/)
- 🧑‍💻 [GitHub Repository](https://github.com/centeio/GraphSD)

---

## 📚 Citation

If you use **GraphSD** in your research, please cite the following publications:

### 📝 Journal Article (Expert Systems, 2023)

> Centeio Jorge, C., Atzmueller, M., Momahed Heravi, B., Gibson, J.L., Rossetti, R.J.F., & Rebelo de Sá, C. (2023).  
> *"Want to come play with me?" Outlier subgroup discovery on spatio-temporal interactions*.  
> Expert Systems, Journal of Knowledge Engineering, 40(5).  
> [https://doi.org/10.1111/exsy.12686](https://doi.org/10.1111/exsy.12686)

```bibtex
@article{DBLP:journals/es/JorgeAHGRS23,
  author  = {Carolina Centeio Jorge and Martin Atzmueller and Behzad Momahed Heravi and
             Jenny L. Gibson and Rosaldo J. F. Rossetti and Cl{'{a}}udio Rebelo de S{'{a}}},
  title   = {"Want to come play with me?" Outlier subgroup discovery on spatio-temporal interactions},
  journal = {Expert Syst. J. Knowl. Eng.},
  volume  = {40},
  number  = {5},
  year    = {2023},
  doi     = {10.1111/EXSY.12686}
}
```

### 📘 Conference Paper (EPIA 2019)

> Centeio Jorge, C., Atzmueller, M., Momahed Heravi, B., Gibson, J.L., Rebelo de Sá, C., & Rossetti, R.J.F. (2019).  
> *Mining Exceptional Social Behaviour*. In *Progress in Artificial Intelligence: EPIA 2019*, Lecture Notes in Computer Science, vol. 11805. Springer.  
> [https://doi.org/10.1007/978-3-030-30244-3_38](https://doi.org/10.1007/978-3-030-30244-3_38)

```bibtex
@inproceedings{DBLP:conf/epia/JorgeAHGSR19,
  author    = {Carolina Centeio Jorge and Martin Atzmueller and Behzad Momahed Heravi and
               Jenny L. Gibson and Cl{'{a}}udio Rebelo de S{'{a}} and Rosaldo J. F. Rossetti},
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

