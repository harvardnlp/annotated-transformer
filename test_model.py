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
model = torch.load("iwslt_49.pt", map_location=torch.device('cpu')).module

# %% jupyter={"source_hidden": true} tags=[]
epoch = 9
model = torch.load("iwslt%.2d.pt" % epoch, map_location=torch.device('cpu')).module


# %%
def check_outputs(model, vocab_src, vocab_tgt, n_examples=15, pad_idx=2, eos_string="</s>"):

    print("Loading data...")
    train_dataloader, valid_dataloader = create_dataloaders(
        torch.device('cpu'),
        batch_size=1,
    )

    for idx in range(1, n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)        
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        print("Source Text (Input)        : " + \
                ' '.join([vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]).replace("\n",""))
        print("Target Text (Ground Truth) : " + \
                ' '.join([vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]).replace("\n",""))
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = ' '.join([vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]).split(eos_string, 1)[0] + eos_string
        print("Model Output               : " +  model_txt.replace("\n",""))
                

check_outputs(model, vocab_src, vocab_tgt)


# %%
