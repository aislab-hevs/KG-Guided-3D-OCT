# Case-Specific Volumetric Lesion Knowledge Graphs with EyeCLIP for Retinal OCT Diagnosis

Code accompanying the ISBI 2027 manuscript:

**Integrating Case-Specific Volumetric Lesion Knowledge Graphs with Vision–Language Representations for Retinal OCT Diagnosis**

The repository follows the experimental organization of the manuscript: dataset-specific Table 2 experiments, EyeCLIP feature extraction, KG construction and Table 3 ablation, and the Table 4 comparative methods.

## Repository structure

```text
ISBI2027_OCT_KG/
│
├── README.md
├── requirements.txt
│
├── datasets/
│   ├── huang/
│   │   ├── run_table2_huang.py
│   │   └── evaluate_huang_test.py
│   │
│   ├── noor/
│   │   ├── prepare_noor.py
│   │   ├── train_structural_proxy.py
│   │   ├── generate_structural_masks.py
│   │   └── run_table2_noor.py
│   │
│   ├── duke/
│   │   ├── prepare_duke.py
│   │   ├── train_structural_proxy.py
│   │   └── run_table2_duke.py
│   │
│   └── fairvision/
│       ├── train_structural_proxy.py
│       └── run_table2_fairvision.py
│
├── eyeclip/
│   ├── extract_noor_eyeclip.py
│   ├── extract_duke_eyeclip.py
│   └── extract_fairvision_eyeclip.py
│
├── kg_construction/
│   ├── build_huang_kg.py
│   ├── extract_huang_trainval_descriptors.py
│   ├── extract_huang_test_descriptors.py
│   ├── build_noor_kg.py
│   ├── extract_noor_descriptors.py
│   ├── build_duke_kg.py
│   ├── extract_duke_descriptors.py
│   ├── build_fairvision_kg.py
│   ├── extract_fairvision_descriptors.py
│   └── run_table3_kg_ablation.py
│
└── comparative_methods/
    ├── prepare_huang_table4.py
    ├── al_zuabi/
    │   └── run_table4_al_zuabi.py
    ├── vlfatrollout/
    │   ├── run_table4_vlfatrollout.py
    │   └── vlfatrollout_model.py
    └── dinov2/
        └── run_table4_dinov2.py
```

Intermediate predictions, checkpoints, descriptors, and result tables are generated locally when the corresponding scripts are executed and are not packaged in the repository.

## Installation

Reference environment:

- Linux + CUDA
- NVIDIA GeForce RTX 4080 SUPER
- Python 3.11
- PyTorch 2.4.1
- torchvision 0.19.1
- CUDA 12.1
- timm 0.9.16

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Raw OCT datasets, third-party repositories, pretrained model weights, CUDA, and GPU drivers are not distributed with this repository.

Machine-specific locations are represented by `PATH_TO_...` entries and should be replaced with local paths. Experimental splits, seeds, model-selection rules, and validation-selected decision-threshold logic should remain unchanged.

# Table 2 — Cross-Cohort Visual, KG, and Visual + KG Evaluation

Table 2 evaluates Visual-only, KG-only, and Visual + KG models on four retinal OCT cohorts: Huang 3D OCT [1], Noor Eye Hospital OCT [7], Duke / Srinivasan OCT [8], and Harvard-FairVision [9].

The visual branch uses frozen EyeCLIP B-scan representations [2]. The structural branch uses the 128-D case-specific volumetric lesion KG descriptor described under Table 3. Fusion is trained from subject-disjoint out-of-fold predictions of the visual and KG branches.

Across the reported experiments, decision thresholds are selected only from validation/development predictions and are frozen before held-out evaluation; test labels are never used for threshold selection.

## Huang 3D OCT

For the Huang cohort, the 3D pathology localizer is trained on training-subject annotations and frozen before graph construction. Construct the KGs using the final nnU-Net predictions:

```bash
python kg_construction/build_huang_kg.py \
  --pred-dir PATH_TO_HUANG_NNUNET_PREDICTIONS \
  --components-dir datasets/huang/results/mass99_components \
  --kg-dir datasets/huang/results/patient_specific_kg
```

Extract the 128-D structural descriptors:

```bash
python kg_construction/extract_huang_trainval_descriptors.py
python kg_construction/extract_huang_test_descriptors.py
```

The Huang Table 2 scripts consume the frozen EyeCLIP slice features generated for the reported experiment.

Run the cross-fitted Table 2 experiment and final held-out evaluation:

```bash
python datasets/huang/run_table2_huang.py
python datasets/huang/evaluate_huang_test.py
```

## Noor Eye Hospital OCT

Prepare the cohort:

```bash
python datasets/noor/prepare_noor.py
```

Extract frozen EyeCLIP features:

```bash
python eyeclip/extract_noor_eyeclip.py
```

Generate the normal-reference structural evidence:

```bash
python datasets/noor/train_structural_proxy.py
python datasets/noor/generate_structural_masks.py
```

Construct the case-specific KGs and descriptors:

```bash
python kg_construction/build_noor_kg.py
python kg_construction/extract_noor_descriptors.py
```

Run the Table 2 experiment:

```bash
python datasets/noor/run_table2_noor.py
```

## Duke / Srinivasan OCT

Prepare the cohort:

```bash
python datasets/duke/prepare_duke.py
```

Extract frozen EyeCLIP features:

```bash
python eyeclip/extract_duke_eyeclip.py
```

Generate the normal-reference structural evidence:

```bash
python datasets/duke/train_structural_proxy.py
```

Construct the case-specific KGs and descriptors:

```bash
python kg_construction/build_duke_kg.py
python kg_construction/extract_duke_descriptors.py
```

Run the Table 2 experiment:

```bash
python datasets/duke/run_table2_duke.py
```

## Harvard-FairVision

The FairVision experiment uses AMD versus no-AMD as the diagnostic target.

Extract frozen EyeCLIP features:

```bash
python eyeclip/extract_fairvision_eyeclip.py
```

Train the no-AMD-reference structural proxy:

```bash
python datasets/fairvision/train_structural_proxy.py
```

Construct the KGs and descriptors:

```bash
python kg_construction/build_fairvision_kg.py
python kg_construction/extract_fairvision_descriptors.py
```

Run the Table 2 experiment:

```bash
python datasets/fairvision/run_table2_fairvision.py
```

# EyeCLIP Visual Representation

The visual branch uses the frozen EyeCLIP ophthalmic vision–language model [2]. The official EyeCLIP repository and pretrained state dictionary are not copied into this repository.

Set the following entries in the EyeCLIP extraction scripts:

```text
PATH_TO_EYECLIP_REPOSITORY
PATH_TO_EYECLIP_CHECKPOINT
```

The EyeCLIP encoder remains frozen during feature extraction. Dataset-specific extraction scripts are provided for Noor, Duke, and FairVision. The Huang Table 2 release consumes the frozen Huang slice embeddings generated for the reported experiment.

# Table 3 — KG Construction and Structural Ablation

Table 3 analyzes the cumulative components of the case-specific volumetric lesion representation on the Huang validation cohort.

## Upstream pathology localization

For the Huang cohort, pathology annotations from training subjects are merged into a single abnormal-tissue foreground and used to train a 3D nnU-Net [3].

Final configuration:

```text
Dataset       : Dataset502_OCTPathologyBinary
Configuration : 3d_fullres
Trainer       : nnUNetTrainer_50epochs
Plans         : nnUNetResEncUNetPlans_16G
Fold          : 0
```

Prediction follows the standard nnU-Net v2 interface:

```bash
nnUNetv2_predict \
  -i PATH_TO_NNUNET_INPUTS \
  -o PATH_TO_HUANG_NNUNET_PREDICTIONS \
  -d 502 \
  -c 3d_fullres \
  -tr nnUNetTrainer_50epochs \
  -p nnUNetResEncUNetPlans_16G \
  -f 0
```

The segmenter is frozen before graph construction. Its predictions provide structural pathology localization only; segmentation is not the diagnostic target.

For the external cohorts, matched lesion annotations are not required. Noor and Duke use normal-reference structural evidence, while FairVision uses no-AMD-reference structural evidence. The same downstream KG construction is then applied.

## Case-specific volumetric KG

Given an ordered OCT volume and its predicted structural-evidence mask:

1. Decompose the 3D evidence mask into 26-connected components [10].
2. Rank components by foreground mass.
3. Retain the smallest component prefix accounting for at least 99% of predicted foreground mass.
4. Represent ordered B-scans as slice nodes.
5. Represent retained 3D connected components as lesion nodes.
6. Add lesion–slice relations when a lesion intersects a B-scan.
7. Add inter-lesion relations when lesion depth intervals overlap.
8. Compute diagnosis-neutral attributes describing lesion burden, geometry, slice occupancy/persistence, shared-slice/depth overlap, and normalized spatial displacement.
9. Aggregate the attributed graph into a 128-D case-level structural descriptor.

Diagnosis labels, patient identifiers, and segmentation-class identities are excluded from the structural representation.

Table 3 evaluates the descriptor cumulatively:

| Structural evidence | Dimension |
|---|---:|
| Lesion burden | 4 |
| + lesion geometry | 74 |
| + lesion–slice relations | 94 |
| + inter-lesion relations | 128 |

After the Huang train/validation descriptors have been generated, run:

```bash
python kg_construction/run_table3_kg_ablation.py
```

# Table 4 — Comparative Visual Methods

Table 4 evaluates the three visual comparison methods on the same subject-disjoint Huang cohort and common validation-selected evaluation protocol.

Prepare the common Huang Table 4 inputs:

```bash
python comparative_methods/prepare_huang_table4.py
```

## Al Zuabi

Run the Al Zuabi comparison reported in Table 4:

```bash
python comparative_methods/al_zuabi/run_table4_al_zuabi.py
```

## VLFATRollout

The local adaptation used for the reported VLFATRollout comparison is included in:

```text
comparative_methods/vlfatrollout/vlfatrollout_model.py
```

Set:

```text
PATH_TO_VLFATROLLOUT_REPOSITORY
```

in that file to the official VLFATRollout source repository, and provide the official Duke pretrained checkpoint through:

```bash
export VLFAT_DUKE_CKPT=/path/to/Duke_VLFATROLLOUT_weights.pth
```

Then run:

```bash
python comparative_methods/vlfatrollout/run_table4_vlfatrollout.py
```

## DINOv2

The DINOv2 comparison uses the official ViT-B/14 representation loaded through the `facebookresearch/dinov2` PyTorch Hub interface [5]:

```bash
python comparative_methods/dinov2/run_table4_dinov2.py
```

The **Visual + KG** result shown in Table 4 is the Huang result produced by the Table 2 pipeline and is not retrained separately.

# User-configurable paths

Depending on the experiment, replace the relevant entries below with local paths:

```text
PATH_TO_HUANG_DATASET
PATH_TO_HUANG_NNUNET_PREDICTIONS
PATH_TO_NNUNET_INPUTS
PATH_TO_HUANG_SUBJECT_SPLIT_CSV

PATH_TO_NOOR_SOURCE_DATASET
PATH_TO_NOOR_CANONICAL_DATASET
PATH_TO_NOOR_CANONICAL_VOLUMES

PATH_TO_DUKE_SRINIVASAN_ROOT
PATH_TO_DUKE_SRINIVASAN_DATASET
PATH_TO_DUKE_CANONICAL_DATASET
PATH_TO_DUKE_CANONICAL_VOLUMES

PATH_TO_FAIRVISION_DATASET
PATH_TO_FAIRVISION_MANIFEST_CSV
PATH_TO_FAIRVISION_INNER_SPLIT_CSV
PATH_TO_FAIRVISION_BASE_TO_META_PREDICTIONS
PATH_TO_FAIRVISION_VAL_PREDICTIONS

PATH_TO_EYECLIP_REPOSITORY
PATH_TO_EYECLIP_CHECKPOINT

PATH_TO_VLFATROLLOUT_REPOSITORY
```

`PATH_TO_HUANG_NNUNET_PREDICTIONS` is supplied as the `--pred-dir` argument to `kg_construction/build_huang_kg.py` rather than edited inside the script.

# References

1. Huang, W. et al. “Comprehensive 3D Optical Coherence Tomography Dataset for AMD and DME: Facilitating Deep-Learning-Based 3D Segmentation.” *Scientific Data*, 2026.
2. Shi, D. et al. “A multimodal visual–language foundation model for computational ophthalmology.” *npj Digital Medicine*, 2025.
3. Isensee, F. et al. “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.” *Nature Methods*, 2021.
4. Oghbaie, M. et al. “VLFATRollout: Fully transformer-based classifier for retinal OCT volumes.” *Computerized Medical Imaging and Graphics*, 2024.
5. Oquab, M. et al. “DINOv2: Learning Robust Visual Features without Supervision.” *Transactions on Machine Learning Research*, 2024.
6. Al Zuabi, B. A. et al. “Automated retinal disease classification from optical coherence tomography images using deep learning.” *International Journal of Retina and Vitreous*, 2026.
7. Rasti, R. et al. “Macular OCT classification using a multi-scale convolutional neural network ensemble.” *IEEE Transactions on Medical Imaging*, 2017.
8. Srinivasan, P. P. et al. “Fully automated detection of diabetic macular edema and dry age-related macular degeneration from optical coherence tomography images.” *Biomedical Optics Express*, 2014.
9. Luo, Y. et al. “FairVision: Equitable deep learning for eye disease screening via fair identity scaling.” arXiv:2310.02492, 2023.
10. He, L., Chao, Y., and Suzuki, K. “Two efficient label-equivalence-based connected-component labeling algorithms for 3-D binary images.” *IEEE Transactions on Image Processing*, 2011.
