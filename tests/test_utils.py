import numpy as np
import pytest
from segformer.utils import recode_labels


@pytest.fixture
def label2id():
    return {
        "corn_soy": 1,
        "developed": 2,
        "forest": 3,
        "pasture": 4,
        "water": 5,
        "wetlands": 6,
        "road": 7,
        "building": 8,
        "background": 0,
    }


@pytest.fixture
def cdl_mapping():
    return {
        "corn_soy": [1, 5, 26],
        "developed": [82, 121, 122, 123, 124],
        "forest": [63, 141, 142, 143],
        "pasture": [176],
        "water": [83, 111],
        "wetlands": [87, 190, 195],
    }


def test_recode_labels(label2id, cdl_mapping):

    shape = (5, 5)
    cdl_labels = np.full(shape, cdl_mapping["forest"][0])
    cdl_labels[0, 0] = cdl_mapping["water"][0]

    road_labels = np.zeros_like(cdl_labels)
    road_labels[2, 3] = 1
    road_labels[4, 4] = 1

    building_labels = np.zeros_like(cdl_labels)
    building_labels[4, 4] = 1

    recoded = recode_labels(
        cdl_labels, road_labels, building_labels, label2id, cdl_mapping,
    )

    assert recoded.shape == cdl_labels.shape

    # This pixel is neither a road nor a building,
    # so the original CDL label remains unchanged
    assert recoded[0, 0] == label2id["water"]

    # The original CDL label is a forest class and we have no buildings or roads in these pixels
    assert recoded[1, 1] == recoded[1, 2] == label2id["forest"]

    # Road labels take precedence over the original CDL labels
    assert recoded[2, 3] == label2id["road"]

    # This pixel is labeled both as a road and a building,
    # but building labels takes precedence over road labels (and the original CDL label)
    assert recoded[4, 4] == label2id["building"]
