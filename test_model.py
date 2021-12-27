# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
execfile("The Annotated Transformer.py")

# %%
GPUtil.showUtilization()

# %%
vocab_tgt.get_itos()[0:10]

# %%
model = torch.load("iwslt_final.pt", map_location=torch.device('cpu')).module

# %% jupyter={"source_hidden": true} tags=[]
epoch = 9
model = torch.load("iwslt%.2d.pt" % epoch, map_location=torch.device('cpu')).module

# %%
train_dataloader, valid_dataloader = create_dataloaders(
    # batch_size=batch_size
    torch.device('cpu'),
    batch_size=1,
)

# %%
pad_idx = 2
b = next(iter(valid_dataloader))
rb = rebatch(pad_idx, b)

print("Source Text (Input):")
print(' '.join([vocab_src.get_itos()[x] for x in rb.src[0] if x != 2]))
print("Target Text (Ground Truth):")
print(' '.join([vocab_tgt.get_itos()[x] for x in rb.tgt_y[0] if x != 2]))
print("Model Output:")
model_out = greedy_decode(model, rb.src, rb.src_mask, 64, 1)[0]
model_txt = ' '.join([vocab_tgt.get_itos()[x] for x in model_out if x != 2])
print(model_txt)

print(rb.src[0])
print(rb.tgt_y[0])
print(model_out)

# %% jupyter={"outputs_hidden": true} tags=[]
model

# %%
vocab_tgt.get_stoi()['.']

# %%
vocab_tgt.get_itos()[344]

# %%
