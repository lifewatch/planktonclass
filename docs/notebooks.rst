3. Notebooks Usage
==================

Overview
--------

This page shows the notebook workflow as a sequence of practical steps.

Use this path when you want an interactive workflow instead of mainly using the CLI or browser API.

Notebook workflow
-----------------

The common order is:

1. install the package
2. create a project
3. validate the config
4. optionally download the pretrained model
5. copy the packaged notebooks into the project
6. work through the notebooks in order
7. inspect training, prediction, and explainability outputs

Step 1: Install the package
---------------------------

.. code-block:: bash

   pip install planktonclass

This installs the Jupyter runtime packages needed to open and execute the notebooks locally.

Step 2: Create a project
------------------------

.. code-block:: bash

   planktonclass init my_project

This creates the standard project structure and a local ``config.yaml``.

Step 3: Validate the config
---------------------------

.. code-block:: bash

   planktonclass validate-config my_project

Step 4: Optional pretrained model
---------------------------------

If you want to start from a published pretrained model:

.. code-block:: bash

   planktonclass pretrained my_project --model FlowCam

Available published pretrained names currently include ``FlowCam``, ``FlowCyto``, and ``PI10``.

Step 5: Copy the notebooks into the project
-------------------------------------------

.. code-block:: bash

   planktonclass notebooks my_project

This creates ``my_project/notebooks/`` and copies the packaged notebooks there.

With the standard project layout from ``planktonclass init``, commands such as ``planktonclass validate-config my_project`` and ``planktonclass train my_project`` automatically use ``my_project/config.yaml``.

To refresh an existing project with updated packaged notebooks:

.. code-block:: bash

   planktonclass notebooks my_project --force

The copied notebooks auto-detect the nearest project ``config.yaml``, so they use the paths inside your local project folder rather than the installed package directory.
They also copy ``data/data_transformation/start``, ``reference_style``, and ``end`` for the image-transformation notebook.

Step 6: Work through the notebooks
----------------------------------

Recommended order:

1. dataset exploration
2. transformations and augmentation
3. model training
4. predictions
5. prediction statistics
6. saliency maps

Notebook list
-------------

``1.0-Dataset_exploration.ipynb``
   Explore class balance, dataset composition, and general dataset statistics.

``1.1-Image_transformation.ipynb``
   Inspect and adapt preprocessing so a new dataset matches the expected training input format.

``1.2-Image_augmentation.ipynb``
   Experiment with augmentation strategies.

``2.0-Model_training.ipynb``
   Run model training interactively.

``3.0-Computing_predictions.ipynb``
   Predict one image or many images and inspect raw outputs.

``3.1-Prediction_statistics.ipynb``
   Evaluate predictions on a labeled split and inspect metrics and confusion-style summaries.

``3.2-Saliency_maps.ipynb``
   Visualize explainability outputs.

Step 7: Important notebook notes
--------------------------------

For ``1.1-Image_transformation.ipynb``:

* put your new raw images in ``data/data_transformation/start/``
* keep one or more reference images in ``data/data_transformation/reference_style/``
* the transformed outputs are written to ``data/data_transformation/end/``

For the model-based notebooks ``3.0-Computing_predictions.ipynb``, ``3.1-Prediction_statistics.ipynb``, and ``3.2-Saliency_maps.ipynb``, the most important variables are ``TIMESTAMP`` and ``MODEL_NAME`` near the top of the notebook. They are prefilled for the published ``FlowCam`` pretrained model so the notebooks run immediately, but you should change them to your own training timestamp and checkpoint name when you want to inspect a newly trained model.

How to open them
----------------

If you are already running Jupyter locally, open the copied project notebook directory and work from there.

If you are inside an AI4OS deployment or a container image that ships the helper commands, you may also have:

.. code-block:: bash

   deep-start -j

That command is deployment-specific. It is not part of the local ``planktonclass`` CLI.
