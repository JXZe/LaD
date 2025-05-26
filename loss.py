import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def _get_batch_logps(logits, labels, average_log_prob):
    assert logits.shape[:-1] == labels.shape, "{}, {}".format(logits.shape[:-1], labels.shape)

    labels = labels.clone()
    # labels = labels[:, 1:].clone()
    # logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def preference_loss(policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps):
    beta = 0.1
    label_smoothing = 0
    reference_free = True

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    return losses, chosen_rewards.detach(), rejected_rewards.detach()

def cal_rpo_loss(logits, ref_logits, labels):
    batch_size = logits.shape[0] // 2
    logps = _get_batch_logps(logits, labels, average_log_prob=False)
    ref_logps = _get_batch_logps(ref_logits, labels, average_log_prob=False)

    chosen_logps = logps[:batch_size]
    rejected_logps = logps[batch_size:]
    chosen_ref_logps = ref_logps[:batch_size]
    rejected_ref_logps = ref_logps[batch_size:]

    losses, chosen_rewards, rejected_rewards = preference_loss(
        chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps)
    return losses.mean()
    
def cal_glm_loss(logits, labels):
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss