import torch
from loss import cal_glm_loss, cal_rpo_loss
from utils import combine_data, generate_rpo_input

def main(model, train_loader):
    for data_glm in train_loader:
        prefix_list = data_glm[0]
        query_list = data_glm[1]
        data_glm = data_glm[2:]
        # (input_ids, input_mask) are for prefix
        # (*_ids_query) are for long-term query
        input_ids, input_mask, decoder_ids, decoder_mask, labels, input_ids_query, mask_ids_query, segment_ids_query, position_ids_query = data_glm
        n_glm = input_ids.shape[0]

        model.eval()
        with torch.no_grad():
            batch_generated_query = model.generate(input_ids, input_mask, input_ids_query, mask_ids_query, segment_ids_query, position_ids_query, beam_size=4)
        data_rpo = generate_rpo_input(prefix_list, batch_generated_query)
        data = combine_data(data_glm, data_rpo)
        
        input_ids, input_mask, decoder_ids, decoder_mask, labels, input_ids_query, mask_ids_query, segment_ids_query, position_ids_query = data
        
        loss = 0
        model.train()
        logits = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=decoder_ids,
            decoder_attention_mask=decoder_mask,
            input_ids_query=input_ids_query,
            attention_mask_query=mask_ids_query,
            token_type_ids_query=segment_ids_query,
            position_ids_query=position_ids_query,
            labels=None)
        
        loss_rpo = cal_rpo_loss(logits[n_glm:], logits[n_glm:], labels[n_glm:])
        loss += loss_rpo
        loss_glm = cal_glm_loss(logits[:n_glm, ...], labels[:n_glm, ...])
        loss += loss_glm
        loss.backward()
        
        
    