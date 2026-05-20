"""
API for the image classification package

Date: September 2018
Last updated: March 2026
Original Author: Ignacio Heredia (CSIC)
Updated and maintained by: Wout Decrop (VLIZ)
Contact: wout.decrop@vliz.be
Github: ai4os-hub / phyto-plankton-classification

Notes: Based on https://github.com/indigo-dc/plant-classification-theano/blob/package/plant_classification/api.py

Descriptions:
The API will use the model files inside ../models/api. If not found it will use the model files of the last trained model.
If several checkpoints are found inside ../models/api/ckpts we will use the last checkpoint.

Warnings:
There is an issue of using Flask with Keras: https://github.com/jrosebr1/simple-keras-rest-api/issues/1
The fix done (using tf.get_default_graph()) will probably not be valid for standalone wsgi container e.g. gunicorn,
gevent, uwsgi.
"""

import builtins
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict
from datetime import datetime
from functools import wraps
from aiohttp.web import HTTPException

# Configure warnings early
from planktonclass import warnings_config
warnings_config.configure_warnings()

class LoadingBar:
    def __init__(self, message="Loading"):
        self.message = message
        self.safe_message = message.encode("ascii", "replace").decode("ascii")
        self.loading = False
        self.thread = None

    def animate(self):
        spinner_chars = ['|', '/', '-', '\\']
        idx = 0
        while self.loading:
            sys.stdout.write(f'\r{self.safe_message} {spinner_chars[idx]}')
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner_chars)
            time.sleep(0.1)
            
    def start(self):
        self.loading = True
        self.thread = threading.Thread(target=self.animate)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.loading = False
        if self.thread:
            self.thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.safe_message) + 10) + '\r')
        sys.stdout.flush()


def _safe_extract_zip(zip_ref, destination):
    destination = os.path.abspath(destination)
    for member in zip_ref.infolist():
        member_path = os.path.abspath(os.path.join(destination, member.filename))
        if os.path.commonpath([destination, member_path]) != destination:
            raise ValueError(f"Unsafe path found in zip archive: {member.filename}")
        if member.is_dir():
            os.makedirs(member_path, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(member_path), exist_ok=True)
        with zip_ref.open(member) as src, open(member_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


_import_loader = LoadingBar("Initializing Phytoplankton Classifier...")
_import_loader.start()

try:
    import numpy as np
    import requests
    import tensorflow as tf
    from deepaas.model.v2.wrapper import UploadedFile
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model
    from webargs import fields
    import logging


    from planktonclass import config, paths, test_utils, utils, model_utils
    from planktonclass.data_utils import (
        load_aphia_ids,
        load_class_info,
        load_class_names,
    )
    from planktonclass.train_runfile import train_fn
finally:
    _import_loader.stop()



logger = logging.getLogger(__name__)
ENV_LOG_LEVEL = os.getenv("API_LOG_LEVEL", default="INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())
logger.setLevel(LOG_LEVEL)


NOW = str("{:%Y_%m_%d_%H_%M_%S}".format(datetime.now()))

from marshmallow import fields, ValidationError
from pathlib import Path

loaded_ts, loaded_ckpt = None, None
graph, model, conf, class_names, class_info, aphia_ids = (
    None,
    None,
    None,
    None,
    None,
    None,
)

# Additional parameters
allowed_extensions = set(
    ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
)  # allow only certain file extensions
top_K = 5  # number of top classes predictions to return


def _list_inference_checkpoints(ckpt_dir):
    """Return inference checkpoints supported by the API."""
    supported_exts = (".keras", ".h5")
    return sorted(
        [name for name in os.listdir(ckpt_dir) if name.endswith(supported_exts)]
    )


def _get_default_checkpoint_name(ckpt_list):
    """Prefer .keras checkpoints when multiple formats are available."""
    keras_ckpts = [name for name in ckpt_list if name.endswith(".keras")]
    if keras_ckpts:
        return keras_ckpts[-1]
    return ckpt_list[-1]


def _list_all_inference_checkpoints(models_dir):
    """Return the unique set of supported checkpoint names across all timestamps."""
    ckpt_names = set()
    timestamp_list = next(os.walk(models_dir))[1]
    current_timestamp = paths.timestamp
    try:
        for timestamp in sorted(timestamp_list):
            paths.timestamp = timestamp
            ckpt_dir = paths.get_checkpoints_dir()
            if not os.path.isdir(ckpt_dir):
                continue
            ckpt_names.update(_list_inference_checkpoints(ckpt_dir))
    finally:
        paths.timestamp = current_timestamp
    return sorted(ckpt_names)


def display_banner():
    """Display ASCII art banner when model is ready."""
    banner = r"""                                                                                                                              
                                     +.                              
                                   +:       :==.                     
                                  %       .#.                        
                                 #:*==*  *=                          
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
                       %@#*++****#********#+***#.                    
                       .%*****. +*********##++*=                     
                        .-##*    *%##%%%%%%#+##:                     
                                +*###%##**+*###                      
                               =++*#+**+++++*###-                    
                             .++*****++++++++*##+                    
                              :+*+#%++++++++*+.                ____________________________
                                  ***  :###-                   |PHYTOPLANKTON CLASSIFIER   |        
                                ::#**.  +**+                   |🔬 Model fully loaded      |       
                               .%@+.: --@@@%                   |🚀 Listening on port 5000  |      
                                       :.                      |___________________________|     
                                                                     
    """
    logger.info(banner)


def load_inference_model(timestamp=None, ckpt_name=None):
    """
    Load a model for prediction.

    Parameters
    ----------
    * timestamp: str
        Name of the timestamp to use. The default is the last timestamp in `./models`.
    * ckpt_name: str
        Name of the checkpoint to use. The default is the last checkpoint in `./models/[timestamp]/ckpts`.
    """
    global loaded_ts, loaded_ckpt
    global graph, model, conf, class_names, class_info, aphia_ids

    # Set the timestamp
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        raise Exception(
            "You have no models in your `./models` folder to be used for inference. "
            "Therefore the API can only be used for training."
        )
    elif timestamp is None:
        timestamp = timestamp_list[-1]
    elif timestamp not in timestamp_list:
        raise ValueError(
            "Invalid timestamp name: {}. Available timestamp names are: {}".format(
                timestamp, timestamp_list
            )
        )
    paths.timestamp = timestamp
    logger.info("✓ Loaded model timestamp: %s", timestamp)

    # Set the checkpoint model to use to make the prediction
    ckpt_list = _list_inference_checkpoints(paths.get_checkpoints_dir())
    if not ckpt_list:
        raise Exception(
            "You have no checkpoints in your `./models/{}/ckpts` folder to be used for inference. ".format(
                timestamp
            )
            + "Therefore the API can only be used for training."
        )
    elif ckpt_name is None:
        ckpt_name = _get_default_checkpoint_name(ckpt_list)
    elif ckpt_name not in ckpt_list:
        raise ValueError(
            "Invalid checkpoint name: {}. Available checkpoint names are: {}".format(
                ckpt_name, ckpt_list
            )
        )
    logger.info("✓ Loaded checkpoint: %s", ckpt_name)

    # Clear the previous loaded model
    tf.keras.backend.clear_session()
    # Load the class names and info
    splits_dir = paths.get_ts_splits_dir()
    class_names = load_class_names(splits_dir=splits_dir)
    aphia_ids = load_aphia_ids(splits_dir)
    class_info = None
    if "info.txt" in os.listdir(splits_dir):
        class_info = load_class_info(splits_dir=splits_dir)
        if len(class_info) != len(class_names):
            warnings.warn(
                """The 'classes.txt' file has a different length than the 'info.txt' file.
            If a class has no information whatsoever you should leave that classes row empty or put a '-' symbol.
            The API will run with no info until this is solved."""
            )
            class_info = None
    if class_info is None:
        class_info = ["" for _ in range(len(class_names))]

    # Load training configuration
    conf_path = os.path.join(paths.get_conf_dir(), "conf.json")
    with open(conf_path) as f:
        conf = json.load(f)
        update_with_saved_conf(conf)

    best_model_name = "best_model.keras"
    best_model_path = os.path.join(paths.get_checkpoints_dir(), best_model_name)
    if (
        conf.get("training", {}).get("use_validation", False)
        and ckpt_name in {"final_model.h5", "final_model.keras"}
        and os.path.exists(best_model_path)
    ):
        logger.info(
            "Switching inference checkpoint from %s to %s because validation-trained runs prefer the best checkpoint.",
            ckpt_name,
            best_model_name,
        )
        ckpt_name = best_model_name

    if ckpt_name == "final_model.keras":
        final_model_path = os.path.join(paths.get_checkpoints_dir(), ckpt_name)
        legacy_final_model_path = os.path.join(
            paths.get_checkpoints_dir(), "final_model.h5"
        )
        if not os.path.exists(final_model_path) and os.path.exists(legacy_final_model_path):
            logger.info(
                "Requested checkpoint %s not found, falling back to legacy checkpoint final_model.h5.",
                ckpt_name,
            )
            ckpt_name = "final_model.h5"

    logger.info("Loading model weights...")
    loader = LoadingBar("✓ Loading model weights...")
    loader.start()
    try:
        # Load the model
        model = load_model(
            os.path.join(paths.get_checkpoints_dir(), ckpt_name),
            custom_objects=utils.get_custom_objects(),
            compile=False,
            # Disable deserialization of training config
        )
    finally:
        loader.stop()

    loaded_ts = timestamp
    loaded_ckpt = ckpt_name
    logger.info("✓ Model fully loaded and ready for inference")
    display_banner()


def update_with_saved_conf(saved_conf):
    """
    Update the default YAML configuration with the configuration saved from training
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        if group in saved_conf.keys():
            for g_key, g_val in sorted(val.items()):
                if g_key in saved_conf[group].keys():
                    g_val["value"] = saved_conf[group][g_key]

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)
    for group, values in saved_conf.items():
        if group not in config.conf_dict:
            config.conf_dict[group] = {}
        for key, value in values.items():
            config.conf_dict[group].setdefault(key, value)


def update_with_query_conf(user_args):
    """
    Update the default YAML configuration with the user's input args from the API query
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            if g_key in user_args:
                raw_value = user_args[g_key]
                if not raw_value:
                    continue  # skip if the value is empty
                try:
                    # Try parsing as JSON
                    g_val["value"] = json.loads(raw_value)
                except json.JSONDecodeError:
                    # Fall back to treating it as a plain string
                    g_val["value"] = raw_value
    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def get_image_files_recursive(directory):
    """
    Recursively find all image files in a directory.
    Returns list of tuples: (full_path, original_filename)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                full_path = os.path.join(root, file)
                image_files.append((full_path, file))
    
    return image_files


def catch_error(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            pred = f(*args, **kwargs)
            return {"status": "OK", "predictions": pred}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return wrap


def catch_localfile_error(file_list):
    # Error catch: Empty query
    if not file_list:
        raise ValueError("Empty query")

    # Error catch: Image format error
    for f in file_list:
        extension = os.path.basename(f.content_type).split("/")[-1]
        # extension = mimetypes.guess_extension(f.content_type)
        if extension not in allowed_extensions:
            raise ValueError(
                "Local image format error: "
                "At least one file is not in a standard image format ({}).".format(
                    allowed_extensions
                )
            )


def warm():
    try:
        load_inference_model()
    except Exception as e:
        logger.debug("Model warm-up: %s", str(e))


def prepare_files(directory):
    """
    Prepare a list of dictionaries with attributes mimicking UploadedFile from image files in the directory.

    :param directory: The directory to search for image files.
    :return: A list of dictionaries with attributes similar to UploadedFile.
    """
    # Get all image files from the directory with given extensions
    extensions = ["*.jpg", "*.png", "*.jpeg"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))

    # Create a list of dictionaries with attributes similar to UploadedFile
    uploaded_files = []
    for file_path in files:
        # Extract the filename from the path
        file_name = os.path.basename(file_path)
        uploaded_files.append(
            UploadedFile(
                name="data",
                filename=file_path,
                content_type="image/jpeg",  # Adjust if necessary based on file type
                original_filename=file_name,
            )
        )
    return uploaded_files


@catch_error
def predict(**args):
    logger.debug("Predict with args: %s", args)
    try:
        if not any([args.get("image"), args.get("zip")]):
            raise Exception(
                "You must provide either 'image' or 'zip' in the payload"
            )

        if args.get("zip"):
            # Check if zip file is provided
            logger.info("▌ Processing ZIP file")
            zip_file = args["zip"]

            # Create a temporary directory to extract the files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the zip file
                with zipfile.ZipFile(
                    zip_file.filename, "r"
                ) as zip_ref:
                    _safe_extract_zip(zip_ref, temp_dir)

                # Recursively find all image files in extracted zip
                image_files = get_image_files_recursive(temp_dir)
                
                if not image_files:
                    raise ValueError("No image files found in the zip archive. "
                                   "Supported formats: jpg, jpeg, png, gif, bmp, tiff, webp")

                # Create UploadedFile objects for each image
                uploaded_files = []
                for file_path, original_filename in image_files:
                    uploaded_files.append(
                        UploadedFile(
                            name="data",
                            filename=file_path,
                            content_type="image/jpeg",
                            original_filename=original_filename,
                        )
                    )

                # Assign the list of files to args["files"]
                args["files"] = uploaded_files
                logger.debug("Found %d image files in zip archive", len(image_files))

                return predict_data(args)
        elif args.get("image"):
            logger.info("▌ Processing single image")
            args["files"] = [
                args["image"]
            ]  # patch until list is available
            return predict_data(args)

    except Exception as err:
        logger.exception("Error in predict endpoint")
        # Sanitize error message for HTTPException (cannot contain newlines)
        error_msg = str(err).replace('\n', ' ').replace('\r', ' ')
        raise HTTPException(reason=error_msg) from err


def predict_data(args):
    """
    Function to predict an image in binary format
    """
    # Check user configuration
    logger.debug("Predict with args: %s", args)
    try:

        update_with_query_conf(args)
        conf = config.conf_dict

        merge = False
        catch_localfile_error(args["files"])

        if (
            loaded_ts != conf["testing"]["timestamp"]
            or loaded_ckpt != conf["testing"]["ckpt_name"]
        ):
            load_inference_model(
                timestamp=conf["testing"]["timestamp"],
                ckpt_name=conf["testing"]["ckpt_name"],
            )
            conf = config.conf_dict
        
        # Ensure preprocess_mode is set based on model name
        if "preprocess_mode" not in conf["model"]:
            modelname = conf["model"].get("modelname", "Phytoplankton_EfficientNetV2B0")
            conf["model"]["preprocess_mode"] = model_utils.model_modes.get(modelname, "tf")
            logger.debug("Set preprocess_mode to: %s for model: %s", conf["model"]["preprocess_mode"], modelname)
        
        # Create a list with the path to the images
        filenames = [f.filename for f in args["files"]]
        original_filenames = [
            f.original_filename for f in args["files"]
        ]

        # with graph.as_default():
        pred_lab, pred_prob = test_utils.predict(
            model=model,
            X=filenames,
            conf=conf,
            top_K=top_K,
            filemode="local",
            merge=merge,
        )

        if merge:
            pred_lab, pred_prob = np.squeeze(pred_lab), np.squeeze(
                pred_prob
            )

        return format_prediction(
            pred_lab, pred_prob, original_filenames
        )
    except Exception as err:
        logger.exception("Error in predict_data function")
        # Sanitize error message for HTTPException (cannot contain newlines)
        error_msg = str(err).replace('\n', ' ').replace('\r', ' ')
        raise HTTPException(reason=error_msg) from err


def get_predictions_dir(CONF):
    file_location = CONF.get("testing", {}).get("file_location", None)
    output_directory = CONF["testing"]["output_directory"]

    if file_location is not None:
        if os.path.exists(file_location):
            os.makedirs(
                os.path.join(
                    os.path.dirname(file_location), "predictions"
                ),
                exist_ok=True,
            )
            return os.path.join(
                os.path.dirname(file_location), "predictions"
            )
    else:
        if output_directory is None:
            # Define your get_timestamped_dir() function accordingly
            return os.path.join(
                paths.get_timestamped_dir(), "predictions"
            )
        else:
            return os.path.join(output_directory)


def format_prediction(labels, probabilities, original_filenames):
    try:
        if aphia_ids is not None:
            pred_aphia_ids = [aphia_ids[i] for i in labels.flatten()]
            pred_aphia_ids = [
                aphia_id.tolist() for aphia_id in pred_aphia_ids
            ]
        else:
            pred_aphia_ids = aphia_ids
        class_index_map = {
            index: class_name
            for index, class_name in enumerate(class_names)
        }
        logger.debug("Labels shape: %s, Class index map size: %s", labels.shape, len(class_index_map))
        
        # Handle 2D array of predictions (N images × top_K predictions)
        pred_lab_names = [[class_index_map[int(label)] for label in row] for row in labels]
        
        pred_prob = probabilities

        pred_dict = {
            "filenames": list(original_filenames),
            "pred_lab": pred_lab_names,
            "pred_prob": pred_prob.tolist(),
            "aphia_ids": pred_aphia_ids,
        }
        conf = config.conf_dict
        ckpt_name = conf["testing"]["ckpt_name"]
        split_name = "test"
        pred_path = os.path.join(
            get_predictions_dir(conf),
            "{}+{}+top{}.json".format(ckpt_name, split_name, top_K),
        )
        with open(pred_path, "w") as outfile:
            json.dump(pred_dict, outfile, sort_keys=True)

        return pred_dict
    except Exception as e:
        logger.exception("Error in format_prediction. labels shape=%s, probabilities shape=%s", 
                        labels.shape if hasattr(labels, 'shape') else 'N/A',
                        probabilities.shape if hasattr(probabilities, 'shape') else 'N/A')
        raise


def get_directory_choices(base_path="/srv/data/"):
    # Get a list of all directories in the base_path
    try:
        directories = [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]
        return directories
    except Exception as e:
        logger.warning("Error accessing directories: %s", str(e))
        return []


def resolve_directory(path):
    """Resolve a user-provided directory using the active config root."""
    if isinstance(path, str):
        path = config.normalize_user_path(path.strip("'\""))
        candidate = Path(path)
    else:
        candidate = Path(path)

    if not candidate.is_absolute():
        candidate = Path(config.CONFIG_ROOT) / candidate

    return candidate.resolve()


def validate_directory(path):
    resolved = resolve_directory(path)
    if not resolved.is_dir():
        raise ValueError(f"{resolved} is not a valid directory")
    return resolved


from pathlib import Path


def train(**args):
    """
    Train an image classifier
    """
    try:
        update_with_query_conf(user_args=args)
        CONF = config.conf_dict
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        # Print training startup banner
        logger.info("="*70)
        logger.info("🚀 Starting Phytoplankton Model Training")
        logger.info("="*70)
        logger.info("▌ Training timestamp: %s", timestamp)
        logger.info("▌ Configuration Parameters:")
        
        # Clear session and validate
        K.clear_session()
        resolved_images_dir = validate_directory(args["images_directory"])
        args["images_directory"] = str(resolved_images_dir)
        CONF["general"]["images_directory"] = str(resolved_images_dir)
        
        # Print config table
        config.print_conf_table(CONF)
        
        logger.info("")
        logger.info("="*70)
        logger.info("▌ Initializing training process...")
        logger.info("="*70)

        train_fn(TIMESTAMP=timestamp, CONF=CONF)

        logger.info("="*70)
        logger.info("✓ Training completed successfully!")
        logger.info("=" *70)
        return {"modelname": timestamp}

    except Exception as err:
        logger.critical("✗ Training failed: %s", str(err), exc_info=True)
        # Sanitize error message for HTTPException (cannot contain newlines)
        error_msg = str(err).replace('\n', ' ').replace('\r', ' ')
        raise ValueError(error_msg) from err


def populate_parser(parser, default_conf):
    """
    Returns a arg-parse like parser.
    """
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val["help"] if ("help" in gg_keys) else ""
            type = (
                getattr(builtins, g_val["type"])
                if ("type" in gg_keys)
                else None
            )
            choices = (
                g_val["choices"] if ("choices" in gg_keys) else None
            )

            # Additional info in help string
            help += (
                "\n"
                + "<font color='#C5576B'> Group name: **{}**".format(
                    str(group)
                )
            )
            if choices:
                help += "\n" + "Choices: {}".format(str(choices))
            if type:
                help += "\n" + "Type: {}".format(g_val["type"])
            help += "</font>"

            # Create arg dict
            opt_args = {
                "load_default": json.dumps(g_val["value"]),
                "metadata": {"description": help},
                "required": False,
            }
            if choices:
                json_choices = [json.dumps(i) for i in choices]
                opt_args["metadata"]["enum"] = json_choices
                opt_args["validate"] = fields.validate.OneOf(
                    json_choices
                )
            parser[g_key] = fields.Str(**opt_args)

    return parser


def get_train_args():
    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict(
        [
            ("general", default_conf["general"]),
            ("model", default_conf["model"]),
            ("training", default_conf["training"]),
            ("monitor", default_conf["monitor"]),
            ("dataset", default_conf["dataset"]),
            ("augmentation", default_conf["augmentation"]),
        ]
    )

    return populate_parser(parser, default_conf)


def get_predict_args():
    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([("testing", default_conf["testing"])])

    # Add options for modelname
    timestamp = default_conf["testing"]["timestamp"]
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        timestamp["value"] = ""
    else:
        timestamp["value"] = timestamp_list[-1]
        timestamp["choices"] = timestamp_list

    # Add options for checkpoint names across all available timestamps.
    # The selected timestamp still determines whether a given checkpoint is valid at runtime.
    ckpt_name = default_conf["testing"]["ckpt_name"]
    ckpt_choices = _list_all_inference_checkpoints(paths.get_models_dir())
    if ckpt_choices:
        if timestamp["value"]:
            current_timestamp = paths.timestamp
            try:
                paths.timestamp = timestamp["value"]
                current_ckpts = _list_inference_checkpoints(paths.get_checkpoints_dir())
            finally:
                paths.timestamp = current_timestamp
        else:
            current_ckpts = []

        if current_ckpts:
            ckpt_name["value"] = _get_default_checkpoint_name(current_ckpts)
        else:
            ckpt_name["value"] = ckpt_choices[0]
        ckpt_name["choices"] = ckpt_choices
    else:
        ckpt_name["value"] = ""

    parser["image"] = fields.Field(
        required=False,
        load_default=None,
        # type="file",
        data_key="image",
        #  location="form",
        metadata={
            "description": "Select the image you want to classify.",
            "type": "file",
            "location": "form",
        },
    )

    parser["zip"] = fields.Field(
        required=False,
        load_default=None,
        #  type="file",
        data_key="zip_data",
        # location="form",
        metadata={
            "description": "Select the ZIP file containing images you want to classify.",
            "type": "file",
            "location": "form",
        },
    )

    return populate_parser(parser, default_conf)


def get_metadata(distribution_name="planktonclass"):
    """
    Function to read metadata
    """

    metadata = {
         "name": config.MODEL_METADATA.get("name"),
            "author": config.MODEL_METADATA.get("authors"),
            "author-email": config.MODEL_METADATA.get(
                "author-emails"
            ),
            "description": config.MODEL_METADATA.get("summary"),
            "license": config.MODEL_METADATA.get("license"),
            "version": config.MODEL_METADATA.get("version"),
            
         
        }

    return metadata


schema = {
    "status": fields.Str(),
    "message": fields.Str(),
    "predictions": fields.Field(),
}
