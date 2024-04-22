import torch

pretrained = torch.load("unimol_tools/unimol_tools/weights/mof_pre_no_h_CORE_MAP_20230505.pt")

print(pretrained.keys())
print(pretrained['model'].keys())

results = {}
for k, v in pretrained['model'].items():
    if "embed_tokens.weight" in k:
        continue
    if "gbf.mul.weight" in k:
        continue
    if "gbf.bias.weight" in k:
        continue
    results[k.replace("unimat.", "")] = v

pretrained["model"] = results
torch.save(pretrained, "unimol_tools/unimol_tools/weights/mof_pre.pt")