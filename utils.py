import torch

def combine_data(data_glm, data_rpo):
    data = []
    for x, y in zip(data_glm, data_rpo):
        data.append(torch.cat([x, y], 0))
    return data

def generate_rpo_input(prefix_list, batch_generated_query):
    toxic_thred = 0.6
    pair_list = []
    for prefix, gen_list in zip(prefix_list, batch_generated_query):
        for query in gen_list:
            score = toxic_expert(query)
            if score > 0.6:
                pair_list.append([prefix, query, '[Reject]'])
            else:
                pair_list.append([prefix, '[Reject]', query])

    data_rpo = format_rpo_input(pair_list)
    return data_rpo
