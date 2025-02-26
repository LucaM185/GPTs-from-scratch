import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

default_xkcd = 0.5
if "humor sans" in ([f.name.lower() for f in fm.fontManager.ttflist]):    
    plt.xkcd(default_xkcd)
    plt.rcParams['font.family'] = 'humor sans'
plt.rcParams['image.cmap'] = 'gray'

class xkcdoff:
    def __enter__(self):
        if "xkcd" in ([f.name for f in fm.fontManager.ttflist]):    
            plt.xkcd(0)
            plt.rcParams['font.family'] = 'humor sans'
        
    def __exit__(self, exc_type, exc_value, traceback):
        if "xkcd" in ([f.name for f in fm.fontManager.ttflist]):    
            plt.xkcd(default_xkcd)
            plt.rcParams['font.family'] = 'humor sans'

def linear(x, y):
    plt.xlabel('Number of hours of study')
    plt.ylabel('Test score')
    plt.title(f'{x.numel()} generated samples of students')

def logistic(x, y):
    # if values are only 1 and 0
    if y.unique().numel() == 2:
        plt.xlabel('Number of hours of study')
        plt.ylabel('Test passed (1) or failed (0)')
        plt.title(f'{x.numel()} generated samples of students')
    else:
        plt.xlabel('Number of hours of study')
        plt.ylabel('Probability of passing the test')
        plt.title(f'{x.numel()} generated samples of students')


def att_visualizations(activations):
    plt.imshow(activations[0].detach().cpu().numpy())
    plt.show()


import os
import torch
import matplotlib.pyplot as plt

att_idx = 0
def save_attention_heatmap(attn_weights, layer_idx, step=0, vis_dir="visualizations", save_ratio=0.01):
    global att_idx
    att_idx += 1
    if att_idx % int(1/save_ratio) == 0:
        os.makedirs(vis_dir, exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_weights.detach().cpu(), cmap='viridis')  # Avg over heads
        plt.title(f"Layer {layer_idx} Attention")
        plt.savefig(f"{vis_dir}/attn_layer{layer_idx}_step{step}.png")
        plt.close()

def save_activations(activations, name, vis_dir="visualizations"):
    os.makedirs(vis_dir, exist_ok=True)
    torch.save(activations, f"{vis_dir}/{name}.pt")

