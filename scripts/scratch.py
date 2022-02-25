#%%
import h5py
path_old = '/home/joelye/user_data/nlb/ndt_old/mc_maze_small.h5'
info_old = h5py.File(path_old)
path_new = '/home/joelye/user_data/nlb/mc_maze_small.h5'
info_new = h5py.File(path_new)
path_old_eval = '/home/joelye/user_data/nlb/ndt_old/eval_data_val.h5'
info_eval = h5py.File(path_old_eval)
#%%
# print(info_old.keys())
print(info_old.keys())
print(info_new.keys())
for key in info_old:
    new_key = key.split('_')
    new_key[1] = 'spikes'
    new_key = '_'.join(new_key)
    if new_key in info_new:
        print(key, (info_old[key][:] == info_new[new_key][:]).all())
    else:
        print(new_key, ' missing')
# print((info_old['train_data_heldin'][:] == info_new['train_spikes_heldin'][:]).all())
# print(info_new['train_spikes_heldin'][:].std())
# print(info_old['train_data_heldout'][:].std())
# print(info_new['train_spikes_heldout'][:].std())
# print(info_old['eval_data_heldin'][:].std())

# print((info_old['eval_data_heldout'][:] == info_new['eval_spikes_heldout'][:]).all())
# print(info_eval['mc_maze_small']['eval_spikes_heldout'][:].std())

#%%
#%%
import torch
scratch_payload = torch.load('scratch_rates.pth')
old_payload = torch.load('old_rates.pth')

print((scratch_payload['spikes'] == old_payload['spikes']).all())
print((scratch_payload['rates'] == old_payload['rates']).all())
print((scratch_payload['labels'] == old_payload['labels']).all())
print(scratch_payload['labels'].sum())
print(old_payload['labels'].sum())
print(scratch_payload['loss'].mean(), old_payload['loss'].mean())