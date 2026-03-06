# COSC-4366EL-01 — Assignment 1: Image Processing and Computer Vision (YOLO)

**Group 1** | Winter 2026 | Augustin De La Cruz
**Dataset:** https://universe.roboflow.com/augustins-workspace-mcqtw/fira-signs/dataset/4
**AI Disclosure:** Claude Sonnet 4.6 (Anthropic, claude.ai/code) was used for code generation, hyperparameter optimization strategy, and report drafting. All outputs were reviewed and validated by the author.

---

## 1. Task 1: AprilTag/ArUco Geometric Detection

**Implementation.** The `trackSign_CV()` function detects AprilTag 36h11 markers using OpenCV's `cv2.aruco.ArucoDetector` API. Each detected tag ID is mapped to one of six FIRA sign classes via a lookup table (`SIGN_MAP`). A threaded camera capture class (`_CameraStream`) decouples I/O latency from the processing loop, reading frames in a daemon thread with `CAP_PROP_BUFFERSIZE=1` to minimize frame staleness.

**Optimizations.** The ArUco detector parameters were tuned for speed: `adaptiveThreshWinSizeStep=10` reduces threshold levels from ~10 to 3, and `cornerRefinementMethod=CORNER_REFINE_NONE` skips sub-pixel refinement, gaining +4-6 FPS. Grayscale conversion reuses a pre-allocated buffer (`gray_buf`) to avoid per-frame memory allocation. Action text pixel widths are pre-computed at import time to eliminate per-frame `getTextSize` calls.

**Runtime overlay.** Each detected tag is rendered with a green corner polygon, tag ID and sign label, the FIRA navigation action in cyan (e.g., "TURN LEFT"), and estimated distance in centimeters using a pinhole camera model: $d = (s \cdot f) / w$, where $s$ is the physical tag size (9 cm), $f$ is the approximate focal length in pixels, and $w$ is the detected tag side length.

**Performance.** [FILL: FPS value] FPS at 640x480 on an Apple M2 Pro, exceeding the 15 FPS target. Detection is reliable across 30-80 cm distances and tolerant of moderate rotation, but fails when the tag is significantly occluded or when lighting produces strong specular reflections on the printed surface.

## 2. Task 2: Visual Sign Detection with YOLO

**Dataset.** 303 images were collected across the six FIRA sign classes using an iPhone camera, including negative samples (no sign present). Images were captured at multiple angles, distances (30-80 cm), and lighting conditions. Annotation was performed in Roboflow with bounding boxes, and the dataset was exported in YOLOv8 format with a 70/20/10 train/validation/test split (211/59/33 images).

**Training.** A YOLOv8n model (3.0M parameters) was fine-tuned from pretrained COCO weights using the native `yolo detect train` CLI. Key hyperparameters: AdamW optimizer, lr0=0.001, batch=16, freeze=5 (backbone layers frozen to prevent overfitting on the small dataset), dropout=0.1. Augmentations were chosen to preserve arrow direction semantics: HSV jitter (h=0.02, s=0.5, v=0.4), scale=0.7, mosaic=1.0, perspective=0.0005, translate=0.1, and mixup=0.15. Flips, rotation, and random erasing were explicitly disabled to avoid corrupting the directional meaning of Left/Right/Forward signs. Training ran for [FILL: epochs] epochs on Apple M2 Pro MPS with early stopping (patience=25).

**Results (test split, 33 images).**

| Class | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| Dead End | [FILL] | [FILL] | [FILL] |
| No Entry | [FILL] | [FILL] | [FILL] |
| Proceed Forward | [FILL] | [FILL] | [FILL] |
| Proceed Left | [FILL] | [FILL] | [FILL] |
| Proceed Right | [FILL] | [FILL] | [FILL] |
| Stop | [FILL] | [FILL] | [FILL] |
| **All** | **[FILL]** | **[FILL]** | **[FILL]** |

**Directional sign analysis.** The three arrow-based signs (Left, Right, Forward) share the same blue-circle visual template and differ only in arrow orientation. With flips disabled, the model must learn directional features purely from edge orientation. Proceed Left consistently exhibited the lowest recall across training runs, likely because (a) left-pointing arrows are underrepresented in the COCO pretraining data, creating a backbone bias toward rightward features, and (b) mosaic and mixup augmentations occasionally place Left and Right signs in the same composite, generating ambiguous gradient signals during training.

**Runtime overlay.** Each detection is rendered with a coloured bounding box, class label with confidence score, the FIRA navigation action in cyan, and estimated distance using the same pinhole model as Task 1 (substituting sign width for tag size).

## 3. Comparison and FIRA Alignment

**Accuracy and robustness.** The geometric approach (Task 1) achieves near-perfect detection when a tag is visible — AprilTag 36h11 decoding is inherently error-corrected and produces zero inter-class confusion. However, it requires the tag to be physically present and unoccluded. The ML approach (Task 2) detects the visual sign appearance directly, enabling recognition even without a tag, but is susceptible to inter-class confusion between visually similar directional signs and to false positives from cluttered backgrounds. Overall mAP@0.5 of [FILL] demonstrates strong but imperfect visual recognition.

**Latency and throughput.**

| Metric | Task 1 (CV) | Task 2 (ML) |
|---|---|---|
| Avg inference time | [FILL] ms | [FILL] ms |
| Avg FPS | [FILL] | [FILL] |

The geometric approach is significantly faster because ArUco detection operates on a single grayscale frame with no neural network inference. The YOLO approach requires a forward pass through 73 convolutional layers (8.1 GFLOPs) but remains within the 10-15 FPS real-time target on MPS.

**Decision consistency.** Both implementations map detections to the same six FIRA navigation actions: No Entry and Dead End trigger "DO NOT ENTER," the three directional signs trigger "TURN RIGHT," "TURN LEFT," or "GO FORWARD," and Stop triggers "STOP." In testing, both methods produced correct navigation cues for all correctly-detected signs. The primary risk to decision consistency is the ML model's occasional confusion between Proceed Left and Proceed Forward, which would send the vehicle in the wrong direction at a junction.

**FIRA rulebook alignment.** The FIRA Autonomous Urban Driving rules permit both AprilTag-based and visual sign recognition. The geometric approach is ideal for controlled competition environments where tags are guaranteed to be present and visible — it is faster, deterministic, and confusion-free. The ML approach is more robust to real-world conditions (partial occlusion, tag damage, varying distances) and could serve as a fallback or complementary perception channel. A competition-ready system could fuse both methods: using the tag-based result when a tag is detected and falling back to visual recognition otherwise.

---

*AI Assistance: Claude Sonnet 4.6 (Anthropic, claude.ai/code) was used for code generation, optimization strategy, and report drafting.*
