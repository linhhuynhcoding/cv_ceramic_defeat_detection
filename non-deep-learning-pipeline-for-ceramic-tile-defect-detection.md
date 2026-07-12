# Non-Deep-Learning Pipeline for Ceramic Tile Defect Detection

## Summary

Build the project as a classical computer vision detection pipeline:

**input raw tile image -> preprocess -> suppress tile grid/grout -> generate defect candidates -> classify candidates -> draw boundary/box overlay**

The dataset is COCO-style with bounding boxes only, not segmentation masks. There are 3 usable defect classes:

- `crack`
- `spalling`
- `exfoliation`

Ignore category id `0` because it is only a parent/supercategory label and has no annotations.

## Recommended Goal

The project should output an image with detected defect regions overlaid as boxes or contours, plus predicted class labels.

Because there are no segmentation masks, use **bounding box detection** as the main evaluation target. Contour overlays can be produced from the classical image-processing candidate masks, but should be treated as approximate boundaries.

## Dataset Facts

Use:

- `2024_ground_wall_tile_dataset_latest.v2i.coco/train`
- `2024_ground_wall_tile_dataset_latest.v2i.coco/valid`

Training annotations:

- `crack`: 14,269 boxes
- `spalling`: 15,392 boxes
- `exfoliation`: 201 boxes

Validation annotations:

- `crack`: 2,460 boxes
- `spalling`: 2,450 boxes
- `exfoliation`: 138 boxes

Important implication: `exfoliation` is rare, so the classifier must use class weighting or oversampling.

## Pipeline

### 1. Data Loading

Load COCO annotations and images.

Implementation modules:

- `src/data.py`
- `src/coco_utils.py`

Responsibilities:

- Read `_annotations.coco.json`
- Map category ids:
  - `1 -> crack`
  - `2 -> exfoliation`
  - `3 -> spalling`
- Ignore category id `0`
- Return image path, image array, ground-truth boxes, and labels

### 2. Preprocessing

For each image:

1. Resize or preserve the existing `640x360` size.
2. Remove black border/background:
   - Convert to grayscale or HSV.
   - Create a valid-tile mask using threshold on brightness.
   - Ignore black image borders during detection.
3. Normalize lighting:
   - Convert to LAB.
   - Apply CLAHE to the L channel.
4. Reduce noise:
   - Use bilateral filter or median blur.
5. Convert to useful channels:
   - grayscale
   - LAB
   - HSV
   - edge map

Recommended algorithms:

- `cv2.cvtColor`
- `cv2.createCLAHE`
- `cv2.bilateralFilter`
- `cv2.medianBlur`

### 3. Grout/Grid Suppression

Tile grout lines are strong distractors, especially for cracks.

Detect and suppress them before candidate generation:

1. Threshold dark pixels in grayscale or HSV.
2. Use morphological opening with long horizontal/vertical kernels.
3. Use `cv2.HoughLinesP` to detect long straight grid lines.
4. Dilate the grout mask slightly.
5. Exclude candidates that mostly overlap the grout mask.

Recommended algorithms:

- adaptive thresholding
- morphological opening/closing
- Hough line transform
- connected components

### 4. Candidate Defect Generation

Generate possible defect regions using classical image processing. Do not classify yet.

#### Crack Candidates

Cracks are thin, dark, irregular, line-like structures.

Use:

- black-hat morphology to enhance dark thin defects
- Canny edges
- adaptive thresholding
- skeletonization if available
- connected components

Candidate filters:

- high aspect ratio
- small average width
- irregular/non-grid orientation
- low overlap with grout mask
- sufficient length

Recommended algorithms:

- `cv2.morphologyEx(..., cv2.MORPH_BLACKHAT)`
- `cv2.Canny`
- `cv2.connectedComponentsWithStats`
- optional `skimage.morphology.skeletonize`

#### Spalling Candidates

Spalling usually appears as small chipped, broken, or dark irregular spots.

Use:

- local color difference in LAB
- texture/edge density
- thresholding for dark or rough patches
- connected components

Candidate filters:

- small to medium area
- irregular shape
- high local contrast
- not a straight line
- often near tile edges or damaged grout boundaries, but do not hard-code that as required

#### Exfoliation Candidates

Exfoliation appears as larger surface peeling/discoloration regions.

Use:

- LAB color anomaly detection
- texture anomaly detection
- local variance/entropy
- region growing or connected components

Candidate filters:

- larger area than spalling
- lower line-likeness than cracks
- strong color/texture difference from surrounding tile

Because exfoliation has very few training samples, start with conservative detection and prioritize recall.

### 5. Feature Extraction

For every candidate region, compute handcrafted features.

Shape features:

- area
- width, height
- aspect ratio
- perimeter
- contour area
- solidity
- extent
- circularity
- skeleton length
- estimated average width

Color features:

- mean/std grayscale
- mean/std LAB channels
- contrast against surrounding patch
- dark-pixel ratio

Texture features:

- edge density
- local binary pattern histogram
- GLCM contrast/homogeneity/energy
- local variance

Position/context features:

- overlap with grout mask
- distance to nearest grout line
- candidate area relative to image area

Recommended libraries:

- `opencv-python`
- `numpy`
- `scikit-image`
- `scikit-learn`

### 6. Classical Classifier

Use the candidate features to classify each candidate into:

- `crack`
- `spalling`
- `exfoliation`
- optional `background/false_positive`

Recommended first model:

- `RandomForestClassifier` or `ExtraTreesClassifier`

Reason:

- works well with mixed handcrafted features
- easier to debug than SVM
- handles nonlinear feature interactions
- provides feature importance

Second model to compare:

- `SVC(kernel="rbf", class_weight="balanced")`

Training setup:

- positive samples from ground-truth boxes
- negative/background samples from candidate regions that do not match any ground-truth box
- class weighting enabled because exfoliation is rare
- oversample exfoliation candidates if needed

### 7. Detection Matching

During training/evaluation, match predicted candidate boxes to ground truth using IoU.

Rules:

- A prediction is correct if:
  - predicted class equals ground-truth class
  - IoU >= `0.3` initially
- Also report IoU >= `0.5` as a stricter metric

Use `0.3` first because classical candidate boxes may be rough.

### 8. Postprocessing

After classification:

1. Remove low-confidence predictions.
2. Apply non-maximum suppression per class.
3. Merge nearby crack fragments if they are aligned and close.
4. Draw:
   - bounding box
   - class label
   - optional contour mask from candidate region

Recommended thresholds:

- classifier confidence >= `0.4` initially
- NMS IoU threshold = `0.3`
- minimum candidate area = tune from validation data
- maximum grout overlap = tune from validation data

### 9. Evaluation

Report both detection and classification quality.

Required metrics:

- precision per class
- recall per class
- F1 per class
- confusion matrix
- false positives per image
- IoU-based detection score at `0.3` and `0.5`

Important focus:

- Crack recall
- Spalling precision
- Exfoliation recall, because it is rare

Also save visual outputs:

- original image
- ground-truth overlay
- predicted overlay
- candidate mask/debug overlay

### 10. Suggested Project Structure

```text
ceramic_tile_defect_detection/
  src/
    data.py
    preprocess.py
    grid.py
    candidates.py
    features.py
    train_classifier.py
    predict.py
    evaluate.py
    visualize.py
  outputs/
    debug/
    predictions/
    reports/
  main.py
```

### 11. Development Order

1. Build COCO loader and visualization of ground-truth boxes.
2. Build preprocessing and grout/grid mask.
3. Build candidate generation for cracks first.
4. Add spalling and exfoliation candidate generation.
5. Extract handcrafted features.
6. Train Random Forest / Extra Trees classifier.
7. Add prediction overlay.
8. Add validation metrics.
9. Tune thresholds using validation split.
10. Write final report with examples, metrics, and failure cases.

## Algorithms To Use

Primary algorithms:

- CLAHE
- thresholding
- adaptive thresholding
- Canny edge detection
- morphological black-hat/top-hat
- morphological opening/closing
- Hough line transform
- connected components
- contour analysis
- handcrafted shape/color/texture features
- Random Forest or Extra Trees classifier
- optional SVM baseline
- non-maximum suppression

Do not use:

- CNN
- YOLO
- Faster R-CNN
- Mask R-CNN
- U-Net
- pretrained deep feature extractors

## Public Interfaces

Command-line scripts should expose these flows:

```bash
python -m src.train_classifier --data 2024_ground_wall_tile_dataset_latest.v2i.coco --model outputs/model.joblib
python -m src.predict --image path/to/image.jpg --model outputs/model.joblib --output outputs/predictions/image.jpg
python -m src.evaluate --data 2024_ground_wall_tile_dataset_latest.v2i.coco --model outputs/model.joblib
```

Prediction output should contain:

```python
{
    "boxes": [[x, y, width, height]],
    "labels": ["crack"],
    "scores": [0.87]
}
```

## Test Cases

Minimum tests/scenarios:

- COCO loader ignores category id `0`
- COCO loader correctly maps `1/2/3` to class names
- preprocessing returns same image size
- black border mask excludes non-tile background
- grid mask detects long grout lines
- candidate generator returns boxes inside image bounds
- feature extractor returns fixed-length numeric vectors
- classifier can train and predict on a small subset
- evaluator computes IoU matching correctly
- prediction script writes an output image

## Acceptance Criteria

The project is successful if:

- It runs without deep learning.
- It accepts a raw tile image.
- It outputs an image with predicted defect overlays.
- It predicts one of `crack`, `spalling`, or `exfoliation`.
- It reports validation precision, recall, and F1 per class.
- It includes visual debug outputs showing preprocessing, grid suppression, candidates, and final predictions.

## Assumptions

- The goal is defect localization with overlay, not only whole-image classification.
- Bounding boxes are acceptable as the main output because the dataset has no segmentation masks.
- Classical ML such as Random Forest, Extra Trees, and SVM is allowed because the restriction is only no deep learning.
- Additional non-DL dependencies such as `numpy`, `scikit-learn`, `scikit-image`, `joblib`, and `matplotlib` are acceptable.
- The first implementation should prioritize a clear, explainable pipeline over maximum accuracy.
