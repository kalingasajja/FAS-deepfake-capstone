# System Architecture: Unified Multimodal Face Authenticity Detection

This document provides a comprehensive architectural breakdown of the **Unified Face Authenticity Detection Pipeline Against Physical and Digital Attacks**, detailing data ingestion, unimodal branches, alignment contracts, late-fusion optimization, and evaluation suites.

---

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    %% Global Styling
    classDef dataset fill:#e8f0fe,stroke:#4285f4,stroke-width:2px,color:#174ea6;
    classDef branch fill:#fef7e0,stroke:#f9ab00,stroke-width:2px,color:#b06000;
    classDef fusion fill:#e6f4ea,stroke:#34a853,stroke-width:2px,color:#137333;
    classDef eval fill:#fce8e6,stroke:#ea4335,stroke-width:2px,color:#c5221f;
    classDef output fill:#f3e8fd,stroke:#a142f4,stroke-width:2px,color:#681da8;

    %% 1. Data Ingestion & Partitioning
    subgraph S1 ["1. Data Ingestion & Leakage-Safe Partitioning"]
        RAW["Raw NFI Dataset (492 Videos)<br/>• 280 Real (Celeb-DF v2)<br/>• 85 Face-Swap (Akool, Deepfaker, Reface...)<br/>• 127 Synthetic (Sora, Kling, Luma, Runway...)"]:::dataset
        MASTER["Master Manifest Catalog<br/>(Video ID, Path, Label, Generator, Confounders)"]:::dataset
        SPLIT["Identity & Scene Clustered Splitter<br/>(70% Train / 15% Val / 15% Test)"]:::dataset
        RULES["Leakage-Safe Constraints:<br/>• Real: Subject ID isolation (id0..id61)<br/>• Face-Swap: Target scene clustering<br/>• Synthetic: Prompt ID clustering"]:::dataset
        
        RAW --> MASTER --> SPLIT --> RULES
    end

    %% 2. Multimodal Specialized Branches
    subgraph S2 ["2. Specialized Tri-Modal Extraction & Detection Branches"]
        B1["Branch 1: Spatial Analysis<br/><b>SigLIP / ViT-Base</b><br/>• Facial landmark cropping<br/>• Blending seam detection<br/>• Texture & boundary artifacts"]:::branch
        B2["Branch 2: Frequency Analysis<br/><b>2D FFT & DCT Transforms</b><br/>• Azimuthal spectral integration<br/>• High-frequency grid residues<br/>• Phase spectrum inconsistencies"]:::branch
        B3["Branch 3: Physiological rPPG<br/><b>pyVHR / POS / LSTM</b><br/>• Facial ROI pulse extraction<br/>• Blood Volume Pulse (BVP)<br/>• Cardiac signal periodicity"]:::branch

        SPLIT -.->|"Cropped Frames"| B1
        SPLIT -.->|"Frequency Spectra"| B2
        SPLIT -.->|"Video Temporal BVP"| B3
    end

    %% 3. Data Contract & Alignment Engine
    subgraph S3 ["3. Contract Verification & Alignment Layer"]
        P1["p_spatial ∈ [0, 1]"]:::branch
        P2["p_frequency ∈ [0, 1]"]:::branch
        P3["p_rppg ∈ [0, 1]"]:::branch

        B1 --> P1
        B2 --> P2
        B3 --> P3

        ALIGN["Validation & Alignment Engine (contract.py)<br/>• Video ID reconciliation<br/>• Score range verification [0.0, 1.0]<br/>• NaN / missing modality handling"]:::fusion
        
        P1 --> ALIGN
        P2 --> ALIGN
        P3 --> ALIGN
    end

    %% 4. Fusion Optimization Engine
    subgraph S4 ["4. Multimodal Fusion Engine"]
        direction TB
        OPT["Simplex Grid Search Calibration<br/>• Parameter sweep: Δ = 0.05 on 2-simplex<br/>• Objective: Maximize ROC-AUC on Val<br/>• Threshold Calibration: Maximize F1-Score"]:::fusion
        
        FORMULA["Calibrated Optimal Late Fusion Formula:<br/><b>p_fused = 0.30·p_spatial + 0.55·p_frequency + 0.15·p_rppg</b><br/>Decision Threshold: τ* = 0.35"]:::fusion

        STAGE2["Learned Intermediate Fusion (Optional Stage 2):<br/>• Gated Multimodal Unit (GMU)<br/>• Score-MLP & Cross-Attention"]:::fusion

        ALIGN --> OPT --> FORMULA
        ALIGN -.->|"Feature Embeddings"| STAGE2
    end

    %% 5. Evaluation & Stress Testing Suite
    subgraph S5 ["5. Rigorous Evaluation & Stress Testing Suite"]
        EVAL["Evaluation Engine (evaluate.py)"]:::eval
        ABL["7-Way Modality Ablation<br/>(Spatial, Freq, rPPG, Pairs, Unified)"]:::eval
        GEN["Per-Generator Generalization<br/>(11 Diverse AI Generators)"]:::eval
        STRESS["Confounder Stress Testing<br/>• Lighting: Well-lit vs Low-light<br/>• Motion: Dynamic vs Static"]:::eval

        FORMULA --> EVAL
        EVAL --> ABL
        EVAL --> GEN
        EVAL --> STRESS
    end

    %% 6. Final Decision & Research Deliverables
    subgraph S6 ["6. Final Classification & Deliverables"]
        PRED["Binary Classification Decision<br/><b>p_fused ≥ 0.35 ➔ FAKE</b><br/><b>p_fused < 0.35 ➔ REAL</b>"]:::output
        METRICS["Benchmark Output:<br/>• Accuracy: 97.01%<br/>• ROC-AUC: 0.9909<br/>• F1-Score: 0.9655<br/>• Precision: 96.55% | Recall: 96.55%"]:::output
        ARTIFACTS["Exported Artifacts:<br/>• fusion_metrics.json<br/>• ablation_results.csv<br/>• per_generator_results.csv<br/>• LATE_FUSION_REPORT.doc / .html"]:::output

        EVAL --> PRED --> METRICS --> ARTIFACTS
    end
```

---

## 2. Modality Weight Allocation & Interaction Architecture

```mermaid
flowchart LR
    classDef sp fill:#e8f0fe,stroke:#4285f4,stroke-width:2px,color:#174ea6;
    classDef fq fill:#e6f4ea,stroke:#34a853,stroke-width:2px,color:#137333;
    classDef rp fill:#fef7e0,stroke:#f9ab00,stroke-width:2px,color:#b06000;
    classDef gate fill:#f3e8fd,stroke:#a142f4,stroke-width:2px,color:#681da8;
    classDef out fill:#fce8e6,stroke:#ea4335,stroke-width:2px,color:#c5221f;

    subgraph Inputs ["Branch Posterior Scores"]
        SP["p_spatial<br/>(SigLIP ViT)"]:::sp
        FQ["p_frequency<br/>(2D FFT / DCT)"]:::fq
        RP["p_rppg<br/>(POS / LSTM)"]:::rp
    end

    subgraph Weights ["Simplex Optimal Weights (w_s + w_f + w_r = 1.0)"]
        W_SP["w_spatial = 0.30<br/>• Texture & visual seams<br/>• Facial boundary check"]:::sp
        W_FQ["w_frequency = 0.55<br/>• Primary Anchor<br/>• 100% Face-Swap catch<br/>• Spectral residue"]:::fq
        W_RP["w_rppg = 0.15<br/>• Biological regularizer<br/>• Pulse continuity<br/>• Resolves edge cases"]:::rp
    end

    subgraph FusionGate ["Score-Level Aggregator"]
        SUM["Weighted Linear Sum<br/>p_fused = Σ (w_i · p_i)"]:::gate
        THRESH["Decision Boundary: τ = 0.35"]:::gate
    end

    subgraph Decision ["Output Verdict"]
        FAKE["AUTHENTIC FAKE<br/>(p_fused ≥ 0.35)"]:::out
        REAL["AUTHENTIC REAL<br/>(p_fused < 0.35)"]:::out
    end

    SP --> W_SP --> SUM
    FQ --> W_FQ --> SUM
    RP --> W_RP --> SUM
    SUM --> THRESH
    THRESH -->|True| FAKE
    THRESH -->|False| REAL
```

---

## 3. Data & Artifact Directory Flow

```mermaid
graph TD
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef code fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c;
    classDef result fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;

    D1["data/master_manifest.json"]:::data --> D2["data/splits.json<br/>data/locked_splits.json"]:::data
    D2 --> D3["data/manifests/<br/>• spatial_manifest.json<br/>• frequency_manifest.json<br/>• rppg_manifest.json"]:::data

    D3 --> C1["Team Branch Inference<br/>(Colab / Local GPUs)"]:::code
    C1 --> D4["data/predictions/<br/>• spatial_test_preds.json<br/>• frequency_test_preds.json<br/>• rppg_test_preds.json"]:::data

    D4 --> C2["run_fusion_pipeline.py<br/>fusion/contract.py<br/>fusion/baseline_fusion.py<br/>fusion/evaluate.py"]:::code

    C2 --> R1["results/<br/>• fusion_metrics.json<br/>• ablation_results.csv<br/>• per_generator_results.csv<br/>• stress_test_results.csv"]:::result
    C2 --> R2["Reports & Documentation<br/>• RESEARCH_PAPER.md<br/>• LATE_FUSION_REPORT.doc<br/>• LATE_FUSION_REPORT.html"]:::result
```

