import torch
import torch.nn.functional as F

def compute_clinical_rewards(logits, targets, task="mortality", ref_logits=None, kl_beta=0.04):
    """
    Computes rewards with KL penalty and Soft Rewards.
    Args:
        logits: [G, B, Seq, Num_Classes] - Current Policy
        targets: [B, Seq] - Ground Truth
        ref_logits: [G, B, Seq, Num_Classes] - Reference Policy (SFT model)
        kl_beta: Strength of the KL penalty (prevents mode collapse)
    Returns:
        total_reward: [G, B, S]
        metrics: dict of scalar values for logging (reward, kl, etc)
    """
    G, B, S, C = logits.shape
    
    # Expand targets to [G, B, S]
    if targets.dim() == 2:
        targets_expanded = targets.unsqueeze(0).expand(G, B, S)
    elif targets.dim() == 3:
        targets_expanded = targets.squeeze(-1).unsqueeze(0).expand(G, B, S)

    # --- 1. Compute KL Divergence (The Anchor) ---
    if C == 2: # Mortality
        current_probs = F.softmax(logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        # KL(P || Q) = sum(P * (log P - log Q))
        kl_div = torch.sum(current_probs * (torch.log(current_probs + 1e-8) - torch.log(ref_probs + 1e-8)), dim=-1)
    else:
        # Phenotyping (Binary KL approx)
        current_probs = torch.sigmoid(logits)
        ref_probs = torch.sigmoid(ref_logits)
        kl_div = (current_probs * (torch.log(current_probs + 1e-8) - torch.log(ref_probs + 1e-8)) +
                  (1-current_probs) * (torch.log(1-current_probs + 1e-8) - torch.log(1-ref_probs + 1e-8)))
        kl_div = kl_div.mean(dim=-1)

    # --- 2. Compute Task Reward (Soft/Probabilistic) ---
    if task == "mortality":
        # Extract prob of the TRUE class
        pos_probs = F.softmax(logits, dim=-1)[..., 1] # [G, B, S]
        
        # Reward: Map probability of correct class to [-1, 1] range
        prob_of_truth = torch.where(targets_expanded == 1, pos_probs, 1.0 - pos_probs)
        task_reward = 2.0 * (prob_of_truth - 0.5) 
        
        # Sharpening Bonus: confident (>0.8) AND correct
        is_confident = (prob_of_truth > 0.8).float()
        sharpening_bonus = 0.5 * is_confident
        task_reward += sharpening_bonus

    elif task == "phenotyping":
        probs = torch.sigmoid(logits)
        targets_float = targets_expanded.float()
        dist = torch.abs(targets_float - probs).mean(dim=-1)
        task_reward = 1.0 - (2.0 * dist) # Maps 0.0 dist to +1, 1.0 dist to -1
        
    # --- 3. Total Reward ---
    total_reward = task_reward - (kl_beta * kl_div)
    
    # Pack metrics for logging
    metrics = {
        "reward/total": total_reward.mean().item(),
        "reward/task": task_reward.mean().item(),
        "reward/kl": kl_div.mean().item()
    }
    
    return total_reward, metrics

def grpo_loss_step(model, ref_model, padded_embeddings, padded_mask, padded_labels, 
                   args, device, G=4, beta=0.04):
    """
    Performs one GRPO update step.
    Returns: 
        final_loss: Scalar tensor
        metrics: Dict of scalars (reward, kl) for logging
    """
    B, Seq, Dim = padded_embeddings.shape
    
    # Repeat inputs
    batch_embeddings_rep = padded_embeddings.repeat_interleave(G, dim=0)
    batch_mask_rep = padded_mask.repeat_interleave(G, dim=0)
    
    # --- 1. Current Policy Forward ---
    model.train() 
    outputs = model(batch_embeddings_rep, batch_mask_rep) 
    
    # --- 2. Reference Policy Forward (No Grad) ---
    with torch.no_grad():
        ref_model.eval()
        ref_outputs = ref_model(batch_embeddings_rep, batch_mask_rep)
    
    # --- 3. Compute Rewards ---
    logits_grouped = outputs.view(G, B, Seq, -1)
    ref_logits_grouped = ref_outputs.view(G, B, Seq, -1)
    
    # We do NOT need gradients flowing through reward computation
    with torch.no_grad():
        rewards, metrics = compute_clinical_rewards(
            logits_grouped, padded_labels, 
            task=args.task, ref_logits=ref_logits_grouped, kl_beta=beta
        )
    
    # --- 4. Advantages (Group Normalization) ---
    mean_rewards = rewards.mean(dim=0, keepdim=True)
    std_rewards = rewards.std(dim=0, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8) 
    advantages_flat = advantages.view(B*G, Seq)
    
    # --- 5. Compute Loss ---
    valid_mask = (padded_labels != -100)
    if valid_mask.dim() == 3: valid_mask = valid_mask.all(dim=-1)
    valid_mask_rep = valid_mask.repeat_interleave(G, dim=0)
    targets_rep = padded_labels.float().repeat_interleave(G, dim=0)
    
    if args.task == "mortality":
        if outputs.shape[-1] == 2:
            outputs_for_loss = outputs[:, :, 1].unsqueeze(-1)
        else:
            outputs_for_loss = outputs
        if targets_rep.dim() == 2: targets_rep = targets_rep.unsqueeze(-1)
        
        bce_loss = F.binary_cross_entropy_with_logits(outputs_for_loss, targets_rep, reduction='none') 
        bce_loss = bce_loss.squeeze(-1) 
        
    elif args.task == "phenotyping":
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets_rep, reduction='none').mean(dim=-1)

    # GRPO Policy Gradient Loss: minimize (loss * -advantage)
    pg_loss = bce_loss * (-advantages_flat)
    pg_loss = pg_loss * valid_mask_rep.float()
    
    final_loss = pg_loss.sum() / (valid_mask_rep.sum() + 1e-8)

    return final_loss, metrics