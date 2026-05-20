# planktonclass: PI10

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19663235.svg)](https://doi.org/10.5281/zenodo.19663235)


**Authors:** [Wout Decrop](https://github.com/woutdecrop) *(VLIZ)* and [Jonas Mortelmans](https://github.com/jonasmortelmansvliz) *(VLIZ)*

This branch is focused on the **PI10 processing workflow** at VLIZ!
It uses the published `planktonclass` package for classification and adds the scripts and configuration needed to process **Pi10 `.tar` files** locally.

For general package usage, training workflows, CLI commands, API usage, notebooks, and full documentation, see:

- https://phyto-plankton-classification.readthedocs.io/en/latest/

For PI10-specific workflow details, see:

- [PI10/readme.md](/PI10/readme.md)
- [PI10/VLIZ-Pi-10_processing.py](/PI10/VLIZ-Pi-10_processing.py)

## PI10 Setup

This branch is intended to **install `planktonclass` as a package** rather than install the package source from this branch.

Create and activate a fresh virtual environment:

```bash
mkdir PI10-processing
cd PI10-processing
python -m venv vpi10
.\vpi10\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Install the package and the extra dependencies needed for the PI10 scripts:

```bash
pip install -r requirements-pi10.txt
```

This installs:

- the published `planktonclass` package
- `scikit-image` for image-region measurements used by the PI10 scripts
- `python-dotenv` for loading `.env` mail settings in the PI10 pipeline

After installation, the main PI10 processing script can be started with:

```bash
python PI10/VLIZ-Pi-10_processing.py
```

If you also want to use `PI10/explore_metrics.py`, install the Spotlight viewer dependency too:

```bash
pip install renumics-spotlight
```

That package is only needed for:

- `from renumics import spotlight`
- `PI10/explore_metrics.py`

## Scope Of This Branch

This README only documents the PI10-specific setup and entry points.
The broader `planktonclass` package documentation, including installation options, training, prediction workflows, notebooks, and deployment guidance, is maintained in the main documentation site:

- https://phyto-plankton-classification.readthedocs.io/en/latest/


## Acknowledgements

If you use this project, please consider citing the original paper for the package:

> Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I., Calatrava, A., & Deneudt, K. (2025). *Automated image classification workflow for phytoplankton monitoring*. **Frontiers in Marine Science, 12**. https://doi.org/10.3389/fmars.2025.1699781

and the model weights:

> Decrop, W., & Mortelmans, J. (2026). *phyto-plankton-classification (v1.0-PI10)*. Zenodo. https://doi.org/10.5281/zenodo.19609980
