import pickle


finetune_dir = "./finetuned_data/ace05_mT5_et_trig_arg_notemp_notype/"

with open('{}/{}_all.pkl'.format(finetune_dir, 'train'), 'rb') as f:
    data = pickle.load(f)
    num_data = len(data["all"])
    # even idx is trigger extraction
    # odd idx is event extraction
    e2e_input = [] # take the input of trigger
    e2e_target = [] # role arguments
    e2e_events = []
    for i in range(num_data):
        if data['input'][i].startswith("TriggerExtract"):
            e2e_input.append(data['input'][i])
        else:
            trig = data['input'][i].split(" <|triggerword")[1]
            e2e_input.append(data['input'][i-1].replace("TriggerExtract", "EventExtract") + " <|triggerword" + trig)
        e2e_target.append(data['target'][i])
        e2e_events.append(data['all'][i])
with open('{}/{}_e2e.pkl'.format(finetune_dir, 'train'), 'wb') as f:
    pickle.dump({
        'input': e2e_input,
        'target': e2e_target,
        'all': e2e_events
    }, f)

# for data_type in ['dev', 'test']:
#     with open('{}/{}_all.pkl'.format(finetune_dir, data_type), 'rb') as f:
#         data = pickle.load(f)
#         num_data = len(data["all"])
#         # even idx is trigger extraction
#         # odd idx is event extraction
#         e2e_input = [data['input'][i] for i in range(num_data) if i%2==0] # take the input of trigger
#         e2e_target = [data['target'][i] for i in range(num_data) if i%2==1] # role arguments
#         e2e_events = [data['all'][i] for i in range(num_data) if i%2==1]
#     with open('{}/{}_e2e.pkl'.format(finetune_dir, data_type), 'wb') as f:
#         pickle.dump({
#             'input': e2e_input,
#             'target': e2e_target,
#             'all': e2e_events
#         }, f)

# with open('{}/dev_e2e.pkl'.format(finetune_dir), 'wb') as f:
#     pickle.dump({
#         'input': dev_inputs,
#         'target': dev_targets,
#         'all': dev_events
#     }, f)