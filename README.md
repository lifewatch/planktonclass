planktonclass: FlowCam
=======================================================

[![Smoke Tests](https://github.com/lifewatch/planktonclass/actions/workflows/tests.yml/badge.svg)](https://github.com/lifewatch/planktonclass/actions/workflows/tests.yml)
[![Integration](https://github.com/lifewatch/planktonclass/actions/workflows/integration.yml/badge.svg)](https://github.com/lifewatch/planktonclass/actions/workflows/integration.yml)
[![PyPI version](https://img.shields.io/pypi/v/planktonclass.svg)](https://pypi.org/project/planktonclass/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/planktonclass.svg)](https://pypi.org/project/planktonclass/)

<table>
  <tr>
    <td valign="top">

**Author:** [Wout Decrop](https://github.com/woutdecrop) (VLIZ) 

**Related publication:**  
[*Automated image classification workflow for phytoplankton monitoring*](https://doi.org/10.3389/fmars.2025.1699781)

**Resources:**
- [Documentation](https://planktonclass.readthedocs.io/en/latest/)
- [PyPI package](https://pypi.org/project/planktonclass/)
- [Package downloads](https://pypi.org/project/planktonclass/)

**Projects:** [iMagine](https://www.imagine-ai.eu/)
  
`planktonclass` is a toolkit for training, evaluating, and serving phytoplankton image classifiers!


It was originally developed for FlowCam data, and has also been retrained or adapted in separate branches for other instruments and datasets:

- [FlowCam / main branch](https://github.com/lifewatch/planktonclass/tree/master)
- [Zooscan branch](https://github.com/lifewatch/planktonclass/tree/zooscan)
- [Cyto branch](https://github.com/lifewatch/planktonclass/tree/cyto)
- [PI10 branch](https://github.com/lifewatch/planktonclass/tree/PI10)

If you want the full repository with Docker, OSCAR, AI4OS, packaged deployment assets, and broader project explanation, see:
- [`phyto-plankton-classification`](https://github.com/ai4os-hub/phyto-plankton-classification)



  </td>
    <td valign="top">

<pre>
                 +.
               +:      :==.
              %      .#.
             #:*==* *=
           -+**+*####.
          +********%%.
         +*******#**#+
      ********#%%####+
        .*====+==::=#%%*
        -%**   --::=-:.
        +=#.   -:::+.
-+*++:  +.     +:::*
:+.  .+- ==:   +::::*
=-    == ::-+*+:::::*##-
.+.  :+-.-====-:::::+%#.
 ===*: :++::::-=:++*#=
  -#. -+**:::=*++**%##+
 .=+-=   ##*:**#*%******=
 .=**+  =*++#************#-
         .++*****++++++++*##+
          :+*+#%++++++++*+.
              ***  :###-
            ::#**.  +**+
           .%@+.: --@@@%
                   :.
</pre>

  </td>
  </tr>
</table>

## Install

Install the default package:

```bash
pip install planktonclass
```

Install with GPU support:

```bash
pip install "planktonclass[gpu]"
```

Supported Python versions: `3.10`, `3.11`, `3.12`

If you want more detail about GPU support on Windows, Linux, or WSL2, jump to [GPU Setup](#gpu-setup).

## Choose Your Path

### 1. I want to train locally

Use:

```bash
planktonclass train my_project
```

This is the best choice if you already know where your image folder is and want a direct local workflow.

### 2. I want to use a browser UI / API

Use:

```bash
planktonclass api my_project
```

Then open:

- `http://127.0.0.1:5000/ui`
- `http://127.0.0.1:5000/api#/`

This is the best choice if you want to interact through the DEEPaaS UI or integrate with an external service.

### 3. I want notebooks

Use:

```bash
planktonclass notebooks my_project
```

This copies the packaged notebooks into `my_project/notebooks/`. It is the best choice for exploration, augmentation experiments, prediction analysis, and explainability.

`pip install planktonclass` installs the package dependencies used by the notebooks, including TensorFlow, plotting, and reporting libraries.

## Quick Start

### Option A: Use it locally
[Read the Docs site](https://planktonclass.readthedocs.io/en/latest/)

```bash
pip install planktonclass
```

Then create a project:

```bash
planktonclass init my_project
```

Or create a runnable demo project:

```bash
planktonclass init my_project --demo
```

*OPTIONAL*: Validate the generated config:

```bash
planktonclass validate-config my_project
```


Local training:

```bash
planktonclass train my_project
```

For a quick smoke test on the demo project:

```bash
planktonclass train my_project --quick
```

*OPTIONAL*: Download a published pretrained model into the project:

```bash
planktonclass pretrained my_project --model FlowCam
```

Available published pretrained model names currently include `FlowCam`, `FlowCyto`, and `PI10`.
Only the actual model directory is extracted into `my_project/models`, even when the downloaded archive contains a
full exported project tree.

*OPTIONAL*: Build an inference Docker image from your trained model run:

```bash
planktonclass docker my_project
```

For the published `FlowCam` pretrained model, the packaged checkpoint is currently
`final_model.h5`. The `FlowCyto` and `PI10` published models are expected to use
`best_model.keras`. New training runs created by `planktonclass train`
save `best_model.keras` when validation is enabled. If you train without validation,
the run saves `final_model.keras` instead.

Report generation after training:

```bash
planktonclass report my_project
```

If you leave out `--timestamp`, `planktonclass report` suggests the most recent run, lists the available timestamps, and lets you choose another one by number.
It also lets you choose between `quick` and `full` mode. `quick` is the default and creates the core figures only; `full` also generates the threshold-based plots in the `results/` subfolders.

### Option B: Use api
[Read the Docs site](https://planktonclass.readthedocs.io/en/latest/)

```bash
pip install planktonclass
```

Then create a project:

```bash
planktonclass init my_project
```


Local API:

```bash
planktonclass api my_project
```

### Option C: I want notebooks

Use the normal install:

```bash
pip install planktonclass
```

Then create a project:

```bash
planktonclass init my_project
```

Copy notebooks into the project:

```bash
planktonclass notebooks my_project
```

In the model-based notebooks (`3.0`, `3.1`, and `3.2`), the first variables to check are `TIMESTAMP` and `MODEL_NAME`. They are prefilled for the published pretrained model so the notebooks work out of the box, but when you want to inspect a model from your own training run you should change those two values first.


## Project Structure

After `planktonclass init`, your project looks like this:

```text
my_project/
  config.yaml
  data/
    images/
    dataset_files/
  models/
  notebooks/
```

### What is required?

The only mandatory input is the image directory:

- `data/images/`
- or the directory pointed to by `images_directory` in `config.yaml`

If `data/dataset_files/` is empty, training can generate dataset splits automatically from the image-folder structure.

If you provide your own dataset metadata files, the expected files are:

- custom-split required: `classes.txt`, `train.txt`
- optional: `val.txt`, `test.txt`, `info.txt`, `aphia_ids.txt`

The split files map image paths to integer labels starting at `0`.

## Configuration

The main user config is a project-local `config.yaml`.

It is created by:

```bash
planktonclass init my_project
```

Most users only need to adjust a small number of fields:

- `general.base_directory`
- `general.images_directory`
- `model.modelname`
- `pretrained.use_pretrained`
- `pretrained.name`
- `pretrained.version`
- `training.epochs`
- `training.batch_size`
- `training.use_validation`
- `training.use_test`
- `monitor.use_tensorboard`

Internal-only values such as model-specific preprocessing are now derived automatically and are not meant to be edited by users.

## Local CLI Workflow

The package installs a `planktonclass` command with these main subcommands:

- `planktonclass init [DIR]`
- `planktonclass init [DIR] --demo`
- `planktonclass validate-config [DIR]`
- `planktonclass train [DIR]`
- `planktonclass report [DIR] [--timestamp TS]`
- `planktonclass api [DIR]`
- `planktonclass docker [DIR]`
- `planktonclass pretrained [DIR]`
- `planktonclass list-models [DIR]`
- `planktonclass notebooks [DIR]`

The `pretrained` command accepts a published model name and version, for example:

```bash
planktonclass pretrained my_project --model FlowCyto --version latest
```

The `list-models` command now shows published pretrained models with extra metadata such as architecture, version, and checkpoint name, while local timestamped runs still appear as plain folder names.

Typical local workflow:

```bash
planktonclass init my_project
planktonclass notebooks my_project
planktonclass validate-config my_project
planktonclass train my_project
planktonclass docker my_project
planktonclass report my_project
```

For a faster package smoke test with the demo data:

```bash
planktonclass init my_project --demo
planktonclass train my_project --quick
planktonclass report my_project
```

## API Workflow

Start the API with:

```bash
planktonclass init my_project
planktonclass api my_project
```

Then open:

- `http://127.0.0.1:5000/ui`
- `http://127.0.0.1:5000/api#/`

You can also start DEEPaaS directly after a repo install:

```powershell
$env:planktonclass_CONFIG = (Resolve-Path .\my_project\config.yaml)
$env:DEEPAAS_V2_MODEL = "planktonclass"
deepaas-run --listen-ip 0.0.0.0
```

Important notes:

- `0.0.0.0` is a bind address, not the browser URL
- open `127.0.0.1` in the browser
- for prediction, the browser UI supports file uploads for `image` and `zip`
- for training, `images_directory` is a path field, so it must point to a folder visible to the machine running the API

## Notebook Workflow


Copy the packaged notebooks into your project with:

```bash
planktonclass init my_project
planktonclass notebooks my_project
```

The copied notebooks auto-detect the nearest project `config.yaml`, so they use the paths inside your local project folder rather than the installed package directory.
They also copy `data/data_transformation/start`, `reference_style`, and `end` for the transformation notebook.

Notebook overview:

- `1.0-Dataset_exploration.ipynb`
- `1.1-Image_transformation.ipynb`
- `1.2-Image_augmentation.ipynb`
- `2.0-Model_training.ipynb`
- `3.0-Computing_predictions.ipynb`
- `3.1-Prediction_statistics.ipynb`
- `3.2-Saliency_maps.ipynb`

For `1.1-Image_transformation.ipynb`:

- put your new raw images in `data/data_transformation/start/`
- keep one or more reference images in `data/data_transformation/reference_style/`
- the transformed outputs are written to `data/data_transformation/end/`

## Outputs

Each training run creates a timestamped folder under `models/`:

```text
models/<timestamp>/
  ckpts/
  conf/
  logs/
  stats/
  dataset_files/
  predictions/
  results/
```

Useful outputs include:

- checkpoints like `best_model.keras`
- `stats.json`
- saved prediction JSON files
- saved test metrics JSON files with top-k accuracy, precision, recall, and F1 summaries
- report images and CSV summaries under `results/`

For a portable inference runtime after training, you can package a selected model run into a Docker image:

```bash
planktonclass docker my_project
```

This builds an image from the local package source and bundles the latest trained timestamp by default.
You can choose a specific run or checkpoint with:

```bash
planktonclass docker my_project --timestamp 2026-04-21_120000 --ckpt-name best_model.keras --tag my-plankton-api:latest
```

To generate performance plots after training:

```bash
planktonclass report my_project
```

If you keep the standard project layout created by `planktonclass init`, these commands automatically use `my_project/config.yaml`. Use `--config PATH` only when your config file lives somewhere else.

## GPU Setup

Use:

```bash
pip install "planktonclass[gpu]"
```

This extra is designed to be friendly across platforms:

- Linux / WSL2: installs TensorFlow with the CUDA runtime dependencies
- Native Windows: adds the DirectML plugin on top of the Windows CPU TensorFlow base
- Other platforms: falls back to the standard install behavior

### Quick checks

After installation, run:

```bash
planktonclass doctor
```

What to look for:

- `TensorFlow runtime: GPU enabled`
  GPU is available and TensorFlow can use it.
- `TensorFlow runtime: GPU unavailable`
  The package installed, but the current environment is CPU-only.

You can also check TensorFlow directly:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If that prints an empty list `[]`, the current environment is not using a GPU.

### GPU support by platform

Python support matrix:

- CPU on Windows: Python `3.10`, `3.11`, `3.12`
- CPU on Linux: Python `3.10`, `3.11`, `3.12`
- GPU on Linux / WSL2: Python `3.10`, `3.11`, `3.12`
- GPU on native Windows: Python `3.10` only

- Linux with NVIDIA GPU
  This is the primary supported GPU path for training and inference on current and future servers.
- WSL2 on Windows with NVIDIA GPU
  This follows the same Linux GPU path and is recommended when you want the most future-proof TensorFlow setup on Windows.
- Native Windows
  GPU support uses DirectML and currently works best with Python 3.10.
  This is useful for local laptop testing, but Linux / WSL2 is still the preferred production path.

### Native Windows GPU

For native Windows GPU inference, use Python 3.10 with DirectML:

```powershell
py -3.10 -m venv ..\g310
..\g310\Scripts\python -m pip install --upgrade pip setuptools wheel
..\g310\Scripts\python -m pip install -e ".[gpu]" --no-build-isolation
```

You can also use the helper script:

```powershell
.\scripts\create_gpu_env.ps1
```

If you see a long-path installation error on Windows, do not create the environment too deep inside nested folders.
Use the helper script above or create a short path such as `..\g310`.

If you need to run newer `.keras` checkpoints in that Windows GPU environment, convert them to `.h5` first:

```powershell
python .\scripts\convert_checkpoint_to_h5.py path\to\best_model.keras
```

### Linux or WSL2 GPU

For Linux GPU servers such as Ubuntu machines with NVIDIA drivers available through `nvidia-smi`, the intended install is simply:

```bash
python3 -m venv ~/planktonclass-gpu
source ~/planktonclass-gpu/bin/activate
python -m pip install --upgrade pip
pip install "planktonclass[gpu]"
```

Or use the helper:

```bash
./scripts/setup_gpu_linux.sh
```

This Linux / WSL2 path is the primary supported GPU route for current and future NVIDIA systems.

### GPU Docker

For GPU-packaged inference containers, build with:

```bash
planktonclass docker my_project --gpu
```

and run with:

```bash
docker run --gpus all -p 5000:5000 my-plankton-api:latest
```

## More Documentation

The full documentation is available here:

- [Read the Docs site](https://planktonclass.readthedocs.io/en/latest/)
- [Documentation entry page](docs/index.rst)

Main documentation pages:

- [Installation](docs/installation.rst)
- [Quickstart](docs/quickstart.rst)
- [API usage](docs/api_usage.rst)
- [Python usage](docs/python_usage.rst)
- [Notebooks](docs/notebooks.rst)
- [Reference](docs/reference.rst)

For Docker, OSCAR, AI4OS, and the broader deployment-oriented repository, see:
- https://github.com/ai4os-hub/phyto-plankton-classification


## Development

Choose this only if you want to work on the package itself.

```bash
git clone https://github.com/lifewatch/planktonclass
cd phyto-plankton-classification
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
pip install -e ".[dev]"
python -m pytest
```

## Acknowledgements

If you use this project, please consider citing:

> Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I., Calatrava, A., & Deneudt, K. (2025). *Automated image classification workflow for phytoplankton monitoring*. **Frontiers in Marine Science, 12**. https://doi.org/10.3389/fmars.2025.1699781
