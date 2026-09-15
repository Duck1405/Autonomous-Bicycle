"""Stable attribute order shared by BDD targets, checkpoints and inference."""
import numpy as np

CATEGORIES = ('single white', 'single yellow', 'single other',
              'double white', 'double yellow', 'double other', 'road curb')
STYLES = ('solid', 'dashed')
ATTRIBUTE_NAMES = tuple('lane/' + name for name in CATEGORIES) + STYLES
NUM_ATTRIBUTES = len(ATTRIBUTE_NAMES)


def encode_attributes(category, attributes):
    # -1 means unannotated, not a negative example.
    target = np.full(NUM_ATTRIBUTES, -1, dtype=np.float32)
    name = category.removeprefix('lane/')
    if name in CATEGORIES:
        target[:len(CATEGORIES)] = 0
        target[CATEGORIES.index(name)] = 1
    style = attributes.get('style')
    if name != 'road curb' and style in STYLES:
        target[len(CATEGORIES):] = 0
        target[len(CATEGORIES) + STYLES.index(style)] = 1
    return target


def decode_attributes(probabilities, points, threshold=0.5):
    probabilities = np.asarray(probabilities)
    scores = dict(zip(ATTRIBUTE_NAMES, map(float, probabilities)))
    category_idx = int(np.argmax(probabilities[:len(CATEGORIES)]))
    category = CATEGORIES[category_idx] if probabilities[category_idx] >= threshold else 'unknown'
    style_idx = int(np.argmax(probabilities[len(CATEGORIES):]))
    style = STYLES[style_idx] if probabilities[len(CATEGORIES)+style_idx] >= threshold else 'unknown'
    if category == 'road curb':
        style = 'not applicable'
    color = category.split()[-1] if category not in ('unknown', 'road curb') else 'unknown'
    # Image-relative majority over ALL valid points; not an ego-lane assignment.
    xs = np.asarray(points)[:, 0]
    left, right = np.count_nonzero(xs < .5), np.count_nonzero(xs > .5)
    side = 'left' if left > right else 'right' if right > left else 'center'
    return {'lane_category': category, 'lane_color': color, 'lane_style': style,
            'image_side': side, 'attribute_scores': scores}
