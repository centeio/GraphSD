# GraphSD
In this repository, we share the code we used for exceptional behaviour discovery.
We adapted an existing subgroup discovery technique to deal with spatio-temporal data, focusing on the study of Subgroup Discovery methods and metrics of social networks analysis. We propose to create digraphs from movement data, to represent the interactions of the subjects, and then apply subgroup discovery to the edges of the digraph. 

More information of the theory behind this code can be found on *Mining Exceptional Social Behaviour (2019)* or *“Want to come play with me?” Outlier Subgroup Discovery on Spatio-Temporal Interactions (2023)* (see citations below).

This repository presents a package, in ``graphsd`` - This package can be installed through ``pip install graph-sd``
On the ``scripts`` directory, several example files present the usages of the package. You can find a guided example on how to use the code in this  [Jupyter Notebook](example.ipynb).



If this repo is useful for your work, please cite:

```
@inproceedings{DBLP:conf/epia/JorgeAHGSR19,
  author       = {Carolina Centeio Jorge and
                  Martin Atzmueller and
                  Behzad Momahed Heravi and
                  Jenny L. Gibson and
                  Cl{\'{a}}udio Rebelo de S{\'{a}} and
                  Rosaldo J. F. Rossetti},
  editor       = {Paulo Moura Oliveira and
                  Paulo Novais and
                  Lu{\'{\i}}s Paulo Reis},
  title        = {Mining Exceptional Social Behaviour},
  booktitle    = {Progress in Artificial Intelligence, 19th {EPIA} Conference on Artificial
                  Intelligence, {EPIA} 2019, Vila Real, Portugal, September 3-6, 2019,
                  Proceedings, Part {II}},
  series       = {Lecture Notes in Computer Science},
  volume       = {11805},
  pages        = {460--472},
  publisher    = {Springer},
  year         = {2019},
  url          = {https://doi.org/10.1007/978-3-030-30244-3\_38},
  doi          = {10.1007/978-3-030-30244-3\_38},
  timestamp    = {Sat, 09 Apr 2022 12:47:00 +0200},
  biburl       = {https://dblp.org/rec/conf/epia/JorgeAHGSR19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

or


```
@article{DBLP:journals/es/JorgeAHGRS23,
  author       = {Carolina Centeio Jorge and
                  Martin Atzmueller and
                  Behzad Momahed Heravi and
                  Jenny L. Gibson and
                  Rosaldo J. F. Rossetti and
                  Cl{\'{a}}udio Rebelo de S{\'{a}}},
  title        = {"Want to come play with me?" Outlier subgroup discovery on spatio-temporal
                  interactions},
  journal      = {Expert Syst. J. Knowl. Eng.},
  volume       = {40},
  number       = {5},
  year         = {2023},
  url          = {https://doi.org/10.1111/exsy.12686},
  doi          = {10.1111/EXSY.12686},
  timestamp    = {Mon, 03 Mar 2025 21:38:11 +0100},
  biburl       = {https://dblp.org/rec/journals/es/JorgeAHGRS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

