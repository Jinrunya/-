from tqdm import tqdm
def final_output(outputs,labelvocab):
    final_outputs=['guid,tag']
    for guid, label in tqdm(outputs, desc='Outputting the file'):
        final_outputs.append((str(guid) + ',' + labelvocab.id_to_label(label)))
    return final_outputs