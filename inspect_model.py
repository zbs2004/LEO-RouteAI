import torch
sd = torch.load('dqn_model_latest.pth', map_location='cpu')
print('state_dict keys:', len(sd))
for k,v in sd.items():
    try:
        print(k, tuple(v.shape))
    except Exception:
        print(k, type(v))
