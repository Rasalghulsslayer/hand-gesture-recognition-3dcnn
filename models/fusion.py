import torch

def fuse_predictions(logits_hrn, logits_lrn):
    """
    Combine les prédictions des deux réseaux.
    Les entrées sont des Logits (pas encore passés par Softmax).
    """
    # 1. Convertir logits en probabilités (Softmax)
    prob_hrn = torch.nn.functional.softmax(logits_hrn, dim=1)
    prob_lrn = torch.nn.functional.softmax(logits_lrn, dim=1)
    
    # 2. Multiplication élément par élément [cite: 68-69]
    fused_prob = prob_hrn * prob_lrn
    
    # 3. Renormalisation (optionnel mais propre)
    fused_prob = fused_prob / fused_prob.sum(dim=1, keepdim=True)
    
    return fused_prob