# Technical Blueprint: Robust `en-to-indic` Architecture for Pulli

**Target**: `pulli` (Tamil Epigraphy, Inscriptional Reconstruction, Tanglish Normalization, and `en-to-indic` Transliteration)  
**Location**: `pulli/docs/VALIMELI_TO_PULLI_BLUEPRINT.md` and `valimeli/docs/VALIMELI_TO_PULLI_BLUEPRINT.md`

---

## 1. Executive Summary & Problem Formulation

In `en-to-indic` transliteration, the model faces a **many-to-one collapse problem**:
$$\begin{rcases}
\text{"vanthu"} \\
\text{"vandhu"} \\
\text{"vantu"} \\
\text{"vanthuu"}
\end{rcases} \quad \xrightarrow{\quad \text{Pulli Engine} \quad} \quad \mathbf{\text{வந்து}} \quad (\text{Canonical Tamil})$$

$$\begin{rcases}
\text{"kandii"} \\
\text{"kontaadi"} \\
\text{"kondaadi"}
\end{rcases} \quad \xrightarrow{\quad \text{Pulli Engine} \quad} \quad \mathbf{\text{கண்டி / கொண்டாடி}}$$

### The Core Finding from ValiMeli:
Crowdworkers and human writers disagree **64.3% of the time on post-nasal stops** and **31.2% on intervocalic stops** when Romanizing Tamil. Standard character-level models treat `nth` and `ndh` as separate character sequences, causing severe fragility on rare roots and inscriptional text.

### The Pulli Architectural Solution:
1. **`PulliAugment`**: Synthesizes all empirical human spelling permutations during training.
2. **`PulliPhoneticContrastiveLoss`**: Enforces InfoNCE contrastive invariance on the Latin encoder so all spelling variants map to an identical latent embedding $\mathbf{z}$.
3. **Target Control Codes**: Uses decoder prefix tokens to deterministically output **Modern Unicode**, **1:1 Epigraphical Readings**, or **Classical Tolkāppiyam Sandhi**.

---

## 2. Module 1: `PulliAugment` (Multi-Spelling Generator)

Add this module to `pulli/src/augment.py`. Use it inside your PyTorch `collate_fn` or dataset pre-processing pipeline to generate on-the-fly Latin spelling permutations.

```python
"""
PulliAugment: Phonologically Grounded Multi-Spelling Data Augmentor for Tamil
Location: pulli/src/augment.py
"""

import random
from typing import List, Set

class PulliAugment:
    # Grounded on measured empirical disagreement rates (Dakshina + Theedhum Nandrum)
    POST_NASAL_MAP = {
        'ந்த': ['nth', 'ndh', 'nd', 'nt'],
        'ண்ட': ['nd', 'nt', 'ndd', 'tt'],
        'ங்க': ['ng', 'ngg', 'nk', 'nq'],
        'ம்ப': ['mb', 'mp', 'mbh'],
        'ஞ்ச': ['nj', 'nch', 'ny', 'ns'],
        'ன்ற': ['ndr', 'nr', 'ntr', 'ndhr']
    }
    
    INTERVOCALIC_MAP = {
        'த': ['th', 'd', 'dh', 't'],
        'ட': ['d', 't', 'th', 'dd'],
        'க': ['g', 'k', 'gh', 'h'],
        'ப': ['b', 'p', 'v', 'bh'],
        'ச': ['s', 'ch', 'c', 'j', 'z'],
        'ற': ['r', 'rr', 'tr', 'dr']
    }
    
    SPECIAL_MAP = {
        'ழ': ['zh', 'z', 'l', 'r', 'll'],
        'ள': ['l', 'll', 'L'],
        'ண': ['n', 'nn', 'N'],
        'ன': ['n', 'nn'],
        'ஞ': ['gn', 'nj', 'ny', 'n']
    }
    
    GEMINATE_MAP = {
        'க்க': ['kk', 'k', 'ck'],
        'ச்ச': ['cch', 'ch', 'sch'],
        'ட்ட': ['tt', 't'],
        'த்த': ['tth', 'th', 'tt'],
        'ப்ப': ['pp', 'p'],
        'ற்ற': ['ttr', 'tr', 'tt']
    }

    @classmethod
    def generate_variants(cls, base_roman: str, indic_word: str, max_variants: int = 8) -> List[str]:
        """
        Takes a canonical base Roman string and its Tamil Unicode word,
        returning a list of natural human spelling variations.
        """
        variants = {base_roman.lower().strip()}
        
        # 1. Post-nasal cluster expansions
        for indic_cluster, roman_opts in cls.POST_NASAL_MAP.items():
            if indic_cluster in indic_word:
                new_vars = set()
                for v in variants:
                    for opt_from in roman_opts:
                        if opt_from in v:
                            for opt_to in roman_opts:
                                new_vars.add(v.replace(opt_from, opt_to))
                variants.update(new_vars)
                
        # 2. Special consonant expansions (ழ/ள/ற/ண)
        for indic_char, roman_opts in cls.SPECIAL_MAP.items():
            if indic_char in indic_word:
                new_vars = set()
                for v in variants:
                    for opt_from in roman_opts:
                        if opt_from in v:
                            for opt_to in roman_opts:
                                new_vars.add(v.replace(opt_from, opt_to))
                variants.update(new_vars)
                
        # 3. Intervocalic expansions
        for indic_char, roman_opts in cls.INTERVOCALIC_MAP.items():
            if indic_char in indic_word:
                new_vars = set()
                for v in variants:
                    for opt_from in roman_opts:
                        if opt_from in v:
                            for opt_to in roman_opts:
                                new_vars.add(v.replace(opt_from, opt_to))
                variants.update(new_vars)
                
        result = list(variants)
        random.shuffle(result)
        return result[:max_variants]
```

---

## 3. Module 2: `PulliPhoneticContrastiveLoss` (Encoder Invariance)

Add this module to `pulli/src/losses.py`. It enforces phonetic invariance in the encoder, pulling different Roman spellings of the same Tamil root together in latent space.

```python
"""
Phonetic Invariance Loss for Pulli Encoder
Location: pulli/src/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PulliPhoneticContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_proj: torch.Tensor, word_type_ids: torch.Tensor) -> torch.Tensor:
        """
        z_proj: [Batch_Size, Dim] (Projected encoder pooled embeddings)
        word_type_ids: [Batch_Size] (Integer ID indicating unique Tamil lemma)
        """
        z_norm = F.normalize(z_proj, dim=-1)
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature
        
        # Binary mask for positive pairs sharing the same Tamil root ID
        labels_eq = torch.eq(word_type_ids.unsqueeze(1), word_type_ids.unsqueeze(0)).float()
        diag_mask = torch.eye(labels_eq.shape[0], device=labels_eq.device)
        positive_mask = labels_eq - diag_mask
        
        # If no positive pairs exist in batch, return 0 loss
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=z_proj.device, requires_grad=True)
            
        exp_sim = torch.exp(sim_matrix) * (1.0 - diag_mask)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss
```

---

## 4. Module 3: Target-Conditioned Multi-Task Decoder

Add these special tokens to `pulli/src/vocab.py` and prepend them to the target training sequence to enable multi-task epigraphical decoding.

### Special Control Tokens:
```python
TASK_CANONICAL_UNICODE = "__canonical__"     # Output: Modern Tamil Unicode (வந்து)
TASK_EPIGRAPHIC_1TO1   = "__epigraphic__"    # Output: Literal stone reading (வ-ந-த)
TASK_CLASSICAL_SANDHI  = "__tolkappiyam__"   # Output: Tolkāppiyam morphophonemic form (வந்தனன்)
TASK_PHONETIC_IPA      = "__ipa__"           # Output: IPA transcription [ʋɐnd̪ɯ]
```

### Formatting Training Targets:
```python
def format_pulli_target(tamil_word: str, task: str = "__canonical__") -> str:
    """Prepends control prefix to ensure deterministic decoding."""
    return f"{task} {tamil_word}"
```

---

## 5. Module 4: Evaluation Benchmark Suite (`PulliBench`)

Add this evaluation script to `pulli/src/evaluate_bench.py`. Evaluates across three decoupled tiers.

```python
"""
Pulli Multi-Tier Evaluation Suite
Location: pulli/src/evaluate_bench.py
"""

import torch

def is_allophonically_equivalent(pred: str, target: str) -> bool:
    """
    Evaluates in-the-wild robustness allowing dialectal homophones (ன/ந, ள/ல, ற/ர).
    """
    if pred.strip() == target.strip():
        return True
        
    def collapse(s: str) -> str:
        return s.replace('ன', 'ந').replace('ள', 'ல').replace('ற', 'ர').replace(' ', '')
        
    return collapse(pred) == collapse(target)

def evaluate_tier(model, dataloader, device, task_prefix="__canonical__"):
    model.eval()
    exact_matches = 0
    allophonic_matches = 0
    total = 0
    
    with torch.no_grad():
        for src_tensor, _, raw_indic, raw_roman in dataloader:
            src_tensor = src_tensor.to(device)
            preds = model.generate(src_tensor, prefix=task_prefix)
            
            for p, ref in zip(preds, raw_indic):
                total += 1
                if p.strip() == ref.strip():
                    exact_matches += 1
                if is_allophonically_equivalent(p, ref):
                    allophonic_matches += 1
                    
    return {
        "exact_match_acc": round((exact_matches / total) * 100, 2),
        "allophonic_equiv_acc": round((allophonic_matches / total) * 100, 2),
        "total_samples": total
    }
```

---

## 6. Implementation Checklist for Pulli Agent

1. [ ] **Copy `PulliAugment`** into `pulli/src/augment.py` and enable multi-spelling generation in the data loader.
2. [ ] **Add `PulliPhoneticContrastiveLoss`** to `pulli/src/losses.py` and set loss weight $\lambda_{\text{contrast}} = 0.10$ in the main training loop.
3. [ ] **Format Training Sets** with prefix tokens (`__canonical__`, `__epigraphic__`, `__tolkappiyam__`).
4. [ ] **Train Model**: Run training with the augmented data and contrastive head.
5. [ ] **Sync Weights**: Save the trained checkpoint to `pulli/models/multilingual_multitask_a3_best.pt`.
