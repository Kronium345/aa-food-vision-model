import pytest

from fridgevision.ssd_decoding import build_decoding

OPTIONS = {
    "numBoxes": 19206, "numClasses": 51, "numCoords": 4, "numKeypoints": 0,
    "xScale": 1.0, "yScale": 1.0, "wScale": 1.0, "hScale": 1.0,
    "applyExponentialOnBoxSize": True, "sigmoidScore": False,
}


def test_build_decoding_finds_outputs_by_shape():
    d = build_decoding(OPTIONS, input_dtype="float32", normalization=(127.5, 127.5),
                       output_shapes=[(1, 19206, 4), (1, 19206, 51)], anchors_file="a.bin")
    assert (d["scoresOutput"], d["boxesOutput"]) == (1, 0)
    assert d["boxOrder"] == "yxhw"
    assert d["input"] == {"dtype": "float32", "mean": 127.5, "std": 127.5}
    assert d["anchorsFile"] == "a.bin"


def test_build_decoding_rejects_unknown_outputs():
    with pytest.raises(ValueError):
        build_decoding(OPTIONS, input_dtype="uint8", normalization=(0, 1),
                       output_shapes=[(1, 19206, 7)], anchors_file="a.bin")


def test_build_decoding_rejects_keypoints():
    with pytest.raises(ValueError):
        build_decoding({**OPTIONS, "numKeypoints": 2}, input_dtype="uint8", normalization=(0, 1),
                       output_shapes=[(1, 19206, 4), (1, 19206, 51)], anchors_file="a.bin")
