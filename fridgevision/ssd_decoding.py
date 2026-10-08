"""Extract what the app needs to decode a Model Maker detector on-device.

Model Maker exports raw SSD outputs (per-anchor class scores + box encodings).
MediaPipe Tasks decodes them using a DETECTOR_METADATA block packed into the
model (fixed anchors + TensorsDecodingOptions). The app runs the model with
react-native-fast-tflite instead, so release.py copies that block into
fridge-labels-vN.json (`detector.decoding`) and the anchors into
fridge-anchors-vN.bin, and lib/fridgeScan/decode.ts decodes the same way
MediaPipe's TensorsToDetectionsCalculator does.

Verified against MediaPipe Tasks' ObjectDetector on MediaPipe's own
efficientdet_lite0 export: raw box encodings are ordered y, x, h, w.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DECODING_FORMAT = "mediapipe-ssd"
# MediaPipe's TensorsToDetectionsCalculator default (reverse_output_order = false).
BOX_ORDER = "yxhw"


def build_decoding(
    options: dict,
    *,
    input_dtype: str,
    normalization: tuple[float, float],
    output_shapes: list[tuple[int, ...]],
    anchors_file: str,
) -> dict:
    """The `detector.decoding` block of fridge-labels-vN.json.

    `options` are MediaPipe's TensorsDecodingOptions fields; `output_shapes`
    are the model's output tensor shapes in output order.
    """
    num_boxes = int(options["numBoxes"])
    num_classes = int(options["numClasses"])
    scores_output = boxes_output = None
    for i, shape in enumerate(output_shapes):
        last = shape[-1] if shape else None
        if last == num_classes and scores_output is None and (last != 4 or boxes_output is not None):
            scores_output = i
        elif last == 4 and boxes_output is None:
            boxes_output = i
    if scores_output is None or boxes_output is None:
        raise ValueError(f"can't tell scores from boxes in output shapes {output_shapes}")
    if int(options.get("numCoords", 4)) != 4 or int(options.get("numKeypoints", 0)) != 0:
        raise ValueError("only plain 4-coordinate boxes without keypoints are supported")
    if input_dtype not in ("uint8", "float32"):
        raise ValueError(f"unsupported input dtype {input_dtype}")

    mean, std = normalization
    return {
        "format": DECODING_FORMAT,
        "anchorsFile": anchors_file,
        "numBoxes": num_boxes,
        "numClasses": num_classes,
        "boxOrder": BOX_ORDER,
        "xScale": float(options["xScale"]),
        "yScale": float(options["yScale"]),
        "wScale": float(options["wScale"]),
        "hScale": float(options["hScale"]),
        "applyExponentialOnBoxSize": bool(options["applyExponentialOnBoxSize"]),
        "sigmoidScore": bool(options["sigmoidScore"]),
        "scoresOutput": scores_output,
        "boxesOutput": boxes_output,
        "input": {"dtype": input_dtype, "mean": float(mean), "std": float(std)},
    }


def _interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # training env (TF 2.15)
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def read_ssd_decoding(model_path: Path, anchors_file: str) -> tuple[dict, int, np.ndarray]:
    """(decoding block, input size, anchors as float32 [N, 4] x_center, y_center, w, h).

    Needs `mediapipe` (training env) for the metadata schema, plus a TFLite
    interpreter (TensorFlow or ai-edge-litert) for tensor shapes.
    """
    from mediapipe.tasks.metadata import object_detector_metadata_schema_py_generated as od
    from mediapipe.tasks.python.metadata import metadata

    meta = json.loads(metadata.MetadataDisplayer.with_model_file(str(model_path)).get_metadata_json())
    subgraph = meta["subgraph_metadata"][0]
    custom = {c["name"]: c for c in subgraph.get("custom_metadata", [])}
    if "DETECTOR_METADATA" not in custom:
        raise ValueError(f"{model_path} has no DETECTOR_METADATA (not a Model Maker detector?)")
    root = od.ObjectDetectorOptionsT.InitFromObj(
        od.ObjectDetectorOptions.GetRootAs(bytearray(custom["DETECTOR_METADATA"]["data"]), 0)
    )
    options = {k: getattr(root.tensorsDecodingOptions, k) for k in vars(root.tensorsDecodingOptions)}
    anchors = np.array(
        [[a.xCenter, a.yCenter, a.width, a.height] for a in root.ssdAnchorsOptions.fixedAnchorsSchema.anchors],
        dtype="<f4",
    )
    if len(anchors) != int(options["numBoxes"]):
        raise ValueError(f"{len(anchors)} anchors but numBoxes={options['numBoxes']}")

    mean, std = 0.0, 1.0
    for unit in subgraph["input_tensor_metadata"][0].get("process_units", []):
        if unit.get("options_type") == "NormalizationOptions":
            mean = float(unit["options"]["mean"][0])
            std = float(unit["options"]["std"][0])

    interpreter = _interpreter(model_path)
    (input_details,) = interpreter.get_input_details()
    input_shape = tuple(int(x) for x in input_details["shape"])
    if len(input_shape) != 4 or input_shape[1] != input_shape[2] or input_shape[3] != 3:
        raise ValueError(f"expected a square RGB input [1, S, S, 3], got {input_shape}")
    output_shapes = [tuple(int(x) for x in o["shape"]) for o in interpreter.get_output_details()]

    decoding = build_decoding(
        options,
        input_dtype=np.dtype(input_details["dtype"]).name,
        normalization=(mean, std),
        output_shapes=output_shapes,
        anchors_file=anchors_file,
    )
    return decoding, input_shape[1], anchors
