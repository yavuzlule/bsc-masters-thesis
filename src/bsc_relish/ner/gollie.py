import sys
sys.path.append("../") # Add the GoLLIE base directory to sys path
import rich
import logging
from src.model.load_model import load_model
import black
import inspect
from jinja2 import Template
import tempfile
from src.tasks.utils_typing import AnnotationList
logging.basicConfig(level=logging.INFO)
from typing import Dict, List, Type

model, tokenizer = load_model(
    inference=True,
    model_weights_name_or_path="HiTZ/GoLLIE-7B",
    quantization=None,
    use_lora=False,
    force_auto_device_map=True,
    use_flash_attention=True,
    torch_dtype="bfloat16"
)