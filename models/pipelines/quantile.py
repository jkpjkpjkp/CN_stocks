import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding, Module, Linear, CrossEntropyLoss
import mlflow
import random

from ..prelude.model import dummyLightning
from ..prelude.data import halfdayData

class _quantile_30min(Dataset, halfdayData):
    def __init__(self, filename):
        super().__init__(filename)
        self.q = np.load('./.results/128th_quantiles_of_1min_ret.npy')
        self.q30 = np.load('./.results/q30.npy')
    
    def __getitem__(self, idx):
        x = self.data[idx * self.seq_len : (idx+1) * self.seq_len]
        y = np.prod(np.lib.stride_tricks.sliding_window_view(x, 30), axis=-1).flatten()
        y = np.searchsorted(self.q30, y)
        x = np.searchsorted(self.q, x)
        return x, y

class quantile_30min(dummyLightning):
    def __init__(self, config, trunk):
        super().__init__(config)
        self.trunk = trunk
        self.train_dataset = _quantile_30min('../data/train.npy')
        self.val_dataset = _quantile_30min('../data/val.npy')
        
        self.emb1 = Embedding(config.vocab_size, config.hidden_dim)
        self.emb30 = Embedding(config.vocab_size, config.hidden_dim)
        self.readout = Linear(config.hidden_dim, config.vocab_size)
    
    def pre_proc(self, x1, x30):
        b = x1.shape[0]
        x1 = self.emb1(x1)
        x30 = self.emb30(x30)
        return x1 + torch.concat((torch.zeros((b, 29, self.config.hidden_dim), device=x30.device, dtype=x30.dtype), x30), dim=1)
    
    def forward(self, x1, x30):
        emb = self.pre_proc(x1, x30)
        x = self.trunk(emb)
        return self.readout(x)
    
    def step(self, batch):
        x, y = batch
        y_hat = self(x[:, :-30], y[:, :-30])
        loss = nn.CrossEntropyLoss()(y_hat.view(-1, 128), y[:, 1:].contiguous().view(-1))
        return {
            'loss': loss,
            'logits': y_hat,
        }
    
    def investigate(self, batch, create_plots=True, output_dir='./investigation_output'):
        """
        Investigate model logits and predictions for the given batch.
        
        Args:
            batch: Input batch to investigate
            create_plots: Whether to create visualization plots
            output_dir: Directory to save plots and results
        
        Returns:
            dict: Investigation results including logits, probabilities, and analysis
        """
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        if create_plots:
            Path(output_dir).mkdir(exist_ok=True)
        
        results = []
        
        for i, x in enumerate(batch):
            # Get model predictions
            d = self.step(x.unsqueeze(0))
            logits = d['logits']
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Calculate expected values and uncertainty metrics
            expected_values = torch.sum(probs * torch.arange(128, device=probs.device).float().unsqueeze(0).unsqueeze(0), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_probs, max_indices = torch.max(probs, dim=-1)
            
            # Store results for this sample
            sample_result = {
                'sample_idx': i,
                'logits': logits.detach().cpu(),
                'probs': probs.detach().cpu(),
                'expected_values': expected_values.detach().cpu(),
                'entropy': entropy.detach().cpu(),
                'max_probs': max_probs.detach().cpu(),
                'max_indices': max_indices.detach().cpu(),
                'loss': d['loss'].detach().cpu()
            }
            results.append(sample_result)
            
            # Print analysis for this sample
            print(f"\n=== Sample {i} Analysis ===")
            print(f"Loss: {sample_result['loss'].item():.4f}")
            print(f"Mean Max Probability: {max_probs.mean().item():.4f}")
            print(f"Mean Entropy: {entropy.mean().item():.4f}")
            print(f"Logits shape: {logits.shape}")
            print(f"Expected values range: [{expected_values.min().item():.2f}, {expected_values.max().item():.2f}]")
            
            if create_plots and i < 3:  # Create plots for first 3 samples
                self._create_sample_plots(sample_result, output_dir, i)
        
        # Create summary analysis
        if create_plots:
            self._create_summary_plots(results, output_dir)
        
        print(f"\nInvestigation complete! Results saved to {output_dir}")
        return results
    
    def _create_sample_plots(self, result, output_dir, sample_idx):
        """Create visualization plots for a single sample."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create logits heatmap
        plt.figure(figsize=(12, 8))
        logits = result['logits'][0]  # Remove batch dimension
        sns.heatmap(logits.numpy(), cmap='viridis', aspect='auto')
        plt.title(f'Sample {sample_idx}: Logit Distribution Over Time')
        plt.xlabel('Vocabulary Index')
        plt.ylabel('Time Step')
        plt.colorbar(label='Logit Value')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_{sample_idx}_logits_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create probability heatmap
        plt.figure(figsize=(12, 8))
        probs = result['probs'][0]
        sns.heatmap(probs.numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
        plt.title(f'Sample {sample_idx}: Probability Distribution Over Time')
        plt.xlabel('Vocabulary Index')
        plt.ylabel('Time Step')
        plt.colorbar(label='Probability')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_{sample_idx}_probs_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create expected values and entropy plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Expected values
        expected_vals = result['expected_values'][0]
        ax1.plot(expected_vals.numpy(), marker='o', markersize=2, linewidth=1)
        ax1.set_title(f'Sample {sample_idx}: Expected Values Over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Expected Value')
        ax1.grid(True, alpha=0.3)
        
        # Entropy
        entropy_vals = result['entropy'][0]
        ax2.plot(entropy_vals.numpy(), marker='o', markersize=2, linewidth=1, color='red')
        ax2.set_title(f'Sample {sample_idx}: Prediction Entropy Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Entropy (bits)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_{sample_idx}_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create max probability distribution
        plt.figure(figsize=(10, 6))
        max_probs = result['max_probs'][0]
        plt.hist(max_probs.numpy(), bins=30, alpha=0.7, color='green')
        plt.title(f'Sample {sample_idx}: Max Probability Distribution')
        plt.xlabel('Max Probability')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_{sample_idx}_max_prob_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_plots(self, results, output_dir):
        """Create summary plots across all samples."""
        import matplotlib.pyplot as plt
        
        # Collect metrics across all samples
        all_losses = [r['loss'].item() for r in results]
        all_mean_max_probs = [r['max_probs'].mean().item() for r in results]
        all_mean_entropies = [r['entropy'].mean().item() for r in results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss distribution
        ax1.hist(all_losses, bins=min(20, len(all_losses)), alpha=0.7, color='blue')
        ax1.set_title('Loss Distribution Across Samples')
        ax1.set_xlabel('Loss')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Mean max probability distribution
        ax2.hist(all_mean_max_probs, bins=min(20, len(all_mean_max_probs)), alpha=0.7, color='green')
        ax2.set_title('Mean Max Probability Distribution')
        ax2.set_xlabel('Mean Max Probability')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Mean entropy distribution
        ax3.hist(all_mean_entropies, bins=min(20, len(all_mean_entropies)), alpha=0.7, color='red')
        ax3.set_title('Mean Entropy Distribution')
        ax3.set_xlabel('Mean Entropy')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Scatter plot: entropy vs max probability
        ax4.scatter(all_mean_entropies, all_mean_max_probs, alpha=0.7, s=50)
        ax4.set_title('Entropy vs Max Probability')
        ax4.set_xlabel('Mean Entropy')
        ax4.set_ylabel('Mean Max Probability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/summary_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics
        with open(f'{output_dir}/summary_stats.txt', 'w') as f:
            f.write("Logit Investigation Summary Statistics\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Number of samples analyzed: {len(results)}\n")
            f.write(f"Mean loss: {np.mean(all_losses):.4f} ± {np.std(all_losses):.4f}\n")
            f.write(f"Mean max probability: {np.mean(all_mean_max_probs):.4f} ± {np.std(all_mean_max_probs):.4f}\n")
            f.write(f"Mean entropy: {np.mean(all_mean_entropies):.4f} ± {np.std(all_mean_entropies):.4f}\n")
            
            f.write("\nModel Behavior Analysis:\n")
            f.write("-" * 25 + "\n")
            if np.mean(all_mean_entropies) < 2.0:
                f.write("✓ Model shows generally confident predictions (low entropy)\n")
            else:
                f.write("⚠ Model shows higher uncertainty in predictions (high entropy)\n")
            
            if np.mean(all_mean_max_probs) > 0.7:
                f.write("✓ Model produces sharp probability distributions\n")
            else:
                f.write("⚠ Model produces more uniform probability distributions\n")
        
        print(f"Summary plots and statistics saved to {output_dir}/")

if __name__ == '__main__':
    from ..prelude.model.config import transformerConfig
    from ..prelude.TM import TM
    config = transformerConfig(
        batch_size=1024
    )
    model = quantile_30min(config, TM(config))
    
    model.fit()
    breakpoint()