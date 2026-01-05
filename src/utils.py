import torch
import gc
import matplotlib.pyplot as plt

def clear_gpu_memory():
    """Clear VRAM to avoid OOM errors"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def plot_training_history(history):
    """Plot Loss and Accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss History')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy History')
    ax2.legend()
    
    plt.show()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")