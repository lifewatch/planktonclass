Installation
============

This page is only about how to install ``planktonclass``.

If you want the first practical workflow after installation, use :doc:`quickstart`.

If you want Docker, AI4OS, OSCAR, or the broader project repository, use the companion repository instead:

* ``phyto-plankton-classification``: https://github.com/ai4os-hub/phyto-plankton-classification

Option A: Install from PyPI
---------------------------

Standard package install:

.. code-block:: bash

   pip install planktonclass

Install with GPU support:

.. code-block:: bash

   pip install "planktonclass[gpu]"

Install with notebook support:

.. code-block:: bash

   pip install "planktonclass[notebooks]"

What this gives you:

* the ``planktonclass`` command-line tool
* local training and reporting
* local DEEPaaS API usage
* packaged notebook export commands
* the Python modules used by the package

If you want more detail about GPU support on Windows, Linux, or WSL2, continue to :ref:`gpu-setup`.

Option B: Development install
-----------------------------

Choose this only if you want to work on the package source itself.

.. code-block:: bash

   git clone https://github.com/lifewatch/planktonclass
   cd planktonclass
   python -m venv .venv
   .venv\Scripts\activate
   pip install -U pip
   pip install -e .

After a repository install, you can also start DEEPaaS directly:

.. code-block:: powershell

   $env:planktonclass_CONFIG = (Resolve-Path .\my_project\config.yaml)
   $env:DEEPAAS_V2_MODEL = "planktonclass"
   deepaas-run --listen-ip 0.0.0.0

Important notes
---------------

* use ``127.0.0.1`` in the browser; ``0.0.0.0`` is only the bind address
* for local notebooks, install ``"planktonclass[notebooks]"``
* for training and API usage, you will usually create a project first with ``planktonclass init my_project``

.. _gpu-setup:

GPU setup
---------

Use:

.. code-block:: bash

   pip install "planktonclass[gpu]"

After installation, run:

.. code-block:: bash

   planktonclass doctor

What to look for:

* ``TensorFlow runtime: GPU enabled`` means TensorFlow can use the GPU
* ``TensorFlow runtime: GPU unavailable`` means the current environment is CPU-only

You can also check TensorFlow directly:

.. code-block:: bash

   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Platform notes:

* Linux with NVIDIA GPU: primary supported GPU path for training and inference
* WSL2 on Windows with NVIDIA GPU: recommended Windows-adjacent path for the most future-proof TensorFlow setup
* Native Windows: GPU support uses DirectML and currently works best with Python 3.10

Native Windows GPU example:

.. code-block:: powershell

   py -3.10 -m venv ..\g310
   ..\g310\Scripts\python -m pip install --upgrade pip setuptools wheel
   ..\g310\Scripts\python -m pip install -e ".[gpu]" --no-build-isolation

Or use the helper script from the repository root:

.. code-block:: powershell

   .\scripts\create_gpu_env.ps1

If you hit a Windows long-path installation error, create the environment in a short path such as ``..\g310``.

Linux or WSL2 GPU example:

.. code-block:: bash

   python3 -m venv ~/planktonclass-gpu
   source ~/planktonclass-gpu/bin/activate
   python -m pip install --upgrade pip
   pip install "planktonclass[gpu]"

Or use the helper:

.. code-block:: bash

   ./scripts/setup_gpu_linux.sh

Next step
---------

After installation, continue with :doc:`quickstart`.
