import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import wandb

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MLPVelocityField(nn.Module):
    """MLP-based velocity field that handles variable input dimensions with domain conditioning"""
    def __init__(self, max_dim=3, hidden_dim=128, num_layers=4, time_embed_dim=32, num_domains=3):
        super().__init__()
        self.max_dim = max_dim
        self.time_embed_dim = time_embed_dim
        self.num_domains = num_domains
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Domain embedding (class conditioning)
        self.domain_embed = nn.Embedding(num_domains, time_embed_dim)
        
        # Main MLP - input is padded to max_dim + time_embed_dim + domain_embed_dim
        layers = []
        input_dim = max_dim + time_embed_dim + time_embed_dim  # Added domain embedding
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        layers.append(nn.Linear(hidden_dim, max_dim))
        
        self.main_mlp = nn.Sequential(*layers)
        
    def forward(self, x, t, target_dim, domain_id):
        """
        x: [batch_size, dim] where dim can be 1, 2 or 3
        t: [batch_size] time values
        target_dim: dimension of the target distribution
        domain_id: [batch_size] domain IDs for conditioning
        """
        batch_size = x.shape[0]
        
        # Pad x to max_dim if needed
        if x.shape[1] < self.max_dim:
            padding = torch.zeros(batch_size, self.max_dim - x.shape[1], device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x
            
        # Time embedding
        t_embed = self.time_mlp(t.unsqueeze(-1))
        
        # Domain embedding
        d_embed = self.domain_embed(domain_id)
        
        # Concatenate and process
        input_tensor = torch.cat([x_padded, t_embed, d_embed], dim=1)
        output = self.main_mlp(input_tensor)
        
        # Return only the relevant dimensions
        return output[:, :target_dim]

class RectifiedFlow:
    """Rectified Flow implementation for multiple distributions with domain conditioning"""
    def __init__(self, model):
        self.model = model
    
    def forward(self, x, target_dim, domain_id):
        """Forward pass for training"""
        batch_size = x.shape[0]
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=x.device)
        
        # Sample noise z1
        z1 = torch.randn_like(x)
        
        # Interpolate: z_t = (1-t) * x + t * z1
        t_expanded = t.view(batch_size, *([1] * (x.dim() - 1)))
        z_t = (1 - t_expanded) * x + t_expanded * z1
        
        # Create domain ID tensor
        domain_ids = torch.full((batch_size,), domain_id, dtype=torch.long, device=x.device)
        
        # Predict velocity
        v_theta = self.model(z_t, t, target_dim, domain_ids)
        
        # Target velocity: z1 - x (direction from data to noise)
        target_v = z1 - x
        
        # MSE loss
        loss = torch.mean((v_theta - target_v) ** 2)
        
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples, dim, domain_id, steps=50, device='cuda'):
        """Sample from the learned distribution for a specific domain"""
        # Start from noise
        z = torch.randn(num_samples, dim, device=device)
        
        # Create domain ID tensor
        domain_ids = torch.full((num_samples,), domain_id, dtype=torch.long, device=device)
        
        dt = 1.0 / steps
        
        for i in range(steps):
            t = torch.ones(num_samples, device=device) * (1 - i * dt)
            v = self.model(z, t, dim, domain_ids)
            z = z - dt * v
            
        return z

# Data generation functions
def generate_double_gaussian_1d(n_samples):
    """Two Gaussians separated in 1D"""
    n1 = n_samples // 2
    n2 = n_samples - n1
    
    # First Gaussian centered at -2
    samples1 = torch.randn(n1, 1) * 0.3 + torch.tensor([[-2.0]])
    
    # Second Gaussian centered at 2  
    samples2 = torch.randn(n2, 1) * 0.3 + torch.tensor([[2.0]])
    
    return torch.cat([samples1, samples2], dim=0)

def generate_four_corners_2d(n_samples):
    """Four small Gaussians at corners of a square"""
    n_per_corner = n_samples // 4
    samples = []
    
    corners = [[-2, -2], [2, -2], [-2, 2], [2, 2]]
    
    for i, (cx, cy) in enumerate(corners):
        n = n_per_corner if i < 3 else n_samples - 3 * n_per_corner
        corner_samples = torch.randn(n, 2) * 0.3 + torch.tensor([cx, cy])
        samples.append(corner_samples)
    
    return torch.cat(samples, dim=0)

def generate_spiral_3d(n_samples):
    """3D spiral distribution"""
    t = torch.linspace(0, 4 * np.pi, n_samples)
    
    x = torch.cos(t) * (1 + t * 0.1)
    y = torch.sin(t) * (1 + t * 0.1)  
    z = t * 0.2
    
    # Add some noise
    noise = torch.randn(n_samples, 3) * 0.1
    samples = torch.stack([x, y, z], dim=1) + noise
    
    return samples

def compute_evaluation_metrics(real_samples, generated_samples):
    """Compute various metrics to evaluate generation quality"""
    real_np = real_samples.cpu().numpy() if torch.is_tensor(real_samples) else real_samples
    gen_np = generated_samples.cpu().numpy() if torch.is_tensor(generated_samples) else generated_samples
    
    metrics = {}
    
    # 1. Wasserstein Distance (Earth Mover's Distance) for each dimension
    wasserstein_distances = []
    for dim in range(real_np.shape[1]):
        wd = wasserstein_distance(real_np[:, dim], gen_np[:, dim])
        wasserstein_distances.append(wd)
    metrics['wasserstein_distance'] = np.mean(wasserstein_distances)
    
    # 2. Maximum Mean Discrepancy (MMD) with RBF kernel
    def rbf_kernel(X, Y, gamma=1.0):
        """RBF kernel for MMD computation"""
        XX = np.sum(X**2, axis=1)[:, None]
        YY = np.sum(Y**2, axis=1)[None, :]
        XY = np.dot(X, Y.T)
        return np.exp(-gamma * (XX - 2*XY + YY))
    
    def mmd_rbf(X, Y, gamma=1.0):
        """Maximum Mean Discrepancy with RBF kernel"""
        K_XX = rbf_kernel(X, X, gamma)
        K_YY = rbf_kernel(Y, Y, gamma)
        K_XY = rbf_kernel(X, Y, gamma)
        
        mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        return max(0, mmd)  # MMD should be non-negative
    
    metrics['mmd'] = mmd_rbf(real_np, gen_np)
    
    # 3. Coverage and Precision using manual nearest neighbor computation
    def find_nearest_distances(X, Y):
        """Find distance to nearest neighbor for each point in X from set Y"""
        distances = cdist(X, Y)
        return np.min(distances, axis=1)
    
    # Coverage (percentage of real samples with nearby generated samples)
    real_to_gen_distances = find_nearest_distances(real_np, gen_np)
    threshold = np.percentile(real_to_gen_distances, 95)  # Use 95th percentile as threshold
    coverage = np.mean(real_to_gen_distances < threshold)
    metrics['coverage'] = coverage
    
    # Precision (percentage of generated samples with nearby real samples)  
    gen_to_real_distances = find_nearest_distances(gen_np, real_np)
    precision = np.mean(gen_to_real_distances < threshold)
    metrics['precision'] = precision
    
    # 4. Mean and std comparison
    real_mean = np.mean(real_np, axis=0)
    gen_mean = np.mean(gen_np, axis=0)
    mean_diff = np.linalg.norm(real_mean - gen_mean)
    metrics['mean_difference'] = mean_diff
    
    real_std = np.std(real_np, axis=0)
    gen_std = np.std(gen_np, axis=0)
    std_diff = np.linalg.norm(real_std - gen_std)
    metrics['std_difference'] = std_diff
    
    return metrics

def plot_distributions(real_samples_list, generated_samples_list, titles, metrics_list=None, save_path=None):
    """Plot real vs generated distributions with evaluation metrics"""
    n_dists = len(real_samples_list)
    fig = plt.figure(figsize=(16, 6 * n_dists))
    
    for i, (real, generated, title) in enumerate(zip(real_samples_list, generated_samples_list, titles)):
        if real.shape[1] == 2:
            # 2D plots
            ax1 = fig.add_subplot(n_dists, 2, 2*i + 1)
            ax1.scatter(real[:, 0], real[:, 1], alpha=0.6, s=20, label='Real')
            ax1.set_title(f'{title} - Real')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(n_dists, 2, 2*i + 2)
            ax2.scatter(generated[:, 0], generated[:, 1], alpha=0.6, s=20, label='Generated', color='orange')
            
            # Add metrics text if provided
            if metrics_list and i < len(metrics_list):
                metrics = metrics_list[i]
                metrics_text = (
                    f"Wasserstein: {metrics['wasserstein_distance']:.3f}\n"
                    f"MMD: {metrics['mmd']:.3f}\n"
                    f"Coverage: {metrics['coverage']:.3f}\n"
                    f"Precision: {metrics['precision']:.3f}\n"
                    f"Mean Diff: {metrics['mean_difference']:.3f}\n"
                    f"Std Diff: {metrics['std_difference']:.3f}"
                )
                ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, 
                        verticalalignment='top', fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax2.set_title(f'{title} - Generated')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        elif real.shape[1] == 3:
            # 3D plots
            ax1 = fig.add_subplot(n_dists, 2, 2*i + 1, projection='3d')
            ax1.scatter(real[:, 0], real[:, 1], real[:, 2], alpha=0.6, s=20, label='Real')
            ax1.set_title(f'{title} - Real')
            ax1.legend()
            
            ax2 = fig.add_subplot(n_dists, 2, 2*i + 2, projection='3d')
            ax2.scatter(generated[:, 0], generated[:, 1], generated[:, 2], alpha=0.6, s=20, label='Generated', color='orange')
            
            # Add metrics text for 3D plots
            if metrics_list and i < len(metrics_list):
                metrics = metrics_list[i]
                metrics_text = (
                    f"Wasserstein: {metrics['wasserstein_distance']:.3f}\n"
                    f"MMD: {metrics['mmd']:.3f}\n"  
                    f"Coverage: {metrics['coverage']:.3f}\n"
                    f"Precision: {metrics['precision']:.3f}\n"
                    f"Mean Diff: {metrics['mean_difference']:.3f}\n"
                    f"Std Diff: {metrics['std_difference']:.3f}"
                )
                # For 3D plots, add text as title extension
                ax2.set_title(f'{title} - Generated\n{metrics_text}', fontsize=10)
            else:
                ax2.set_title(f'{title} - Generated')
            ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def train_multi_domain_rf(domains_to_train=[0, 1, 2], epochs=1000, batch_size=256, lr=1e-3, 
                         hidden_dim=128, num_layers=4, log_wandb=True):
    """Train rectified flow on selected domains with wandb logging"""
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLPVelocityField(max_dim=3, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    rf = RectifiedFlow(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Data generators
    generators = [generate_double_gaussian_1d, generate_four_corners_2d, generate_spiral_3d]
    dims = [2, 2, 3]
    names = ['Double Gaussian 2D', 'Four Corners 2D', 'Spiral 3D']
    
    # Setup wandb config
    if log_wandb:
        config = {
            'domains_to_train': domains_to_train,
            'domain_names': [names[i] for i in domains_to_train],
            'num_domains': len(domains_to_train),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'model_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device': str(device),
            'total_dimensions': sum(dims[i] for i in domains_to_train),
            'has_2d_domains': any(dims[i] == 2 for i in domains_to_train),
            'has_3d_domains': any(dims[i] == 3 for i in domains_to_train),
            'mixed_dimensions': len(set(dims[i] for i in domains_to_train)) > 1
        }
        
        # Create a meaningful run name
        domain_str = "_".join([f"D{i}" for i in domains_to_train])
        run_name = f"RF_{domain_str}_{len(domains_to_train)}domains_h{hidden_dim}_l{num_layers}"
        
        wandb.init(
            project="toy_multi_dist_flower",
            name=run_name,
            config=config,
            tags=[
                f"{len(domains_to_train)}_domains",
                f"hidden_{hidden_dim}",
                f"layers_{num_layers}",
                "multi_domain" if len(domains_to_train) > 1 else "single_domain",
                "mixed_dim" if config['mixed_dimensions'] else "same_dim"
            ]
        )
    
    print(f"Training on domains: {[names[i] for i in domains_to_train]}")
    
    # Training loop
    losses = []
    domain_losses = {i: [] for i in domains_to_train}
    
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0
        epoch_domain_losses = {i: 0 for i in domains_to_train}
        n_batches = 0
        
        for domain_id in domains_to_train:
            # Generate batch for this domain
            samples = generators[domain_id](batch_size).to(device)
            
            # Forward pass
            loss = rf.forward(samples, dims[domain_id], domain_id)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_domain_losses[domain_id] = loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # Store per-domain losses
        for domain_id in domains_to_train:
            domain_losses[domain_id].append(epoch_domain_losses[domain_id])
        
        # Log to wandb
        if log_wandb:
            log_dict = {
                'epoch': epoch,
                'loss/total': avg_loss,
                'loss/learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # Log per-domain losses
            for domain_id in domains_to_train:
                log_dict[f'loss/domain_{domain_id}_{names[domain_id].replace(" ", "_")}'] = epoch_domain_losses[domain_id]
            
            wandb.log(log_dict)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
    
    return rf, losses, names, dims, generators, domain_losses

def evaluate_model(rf, names, dims, generators, domains_to_train, n_samples=1000, log_wandb=True):
    """Evaluate the trained model with comprehensive metrics and wandb logging"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    real_samples_list = []
    generated_samples_list = []
    eval_titles = []
    metrics_list = []
    
    print("\nComputing evaluation metrics...")
    
    # Overall metrics aggregation
    all_metrics = {
        'wasserstein_distance': [],
        'mmd': [],
        'coverage': [],
        'precision': [],
        'mean_difference': [],
        'std_difference': []
    }
    
    for domain_id in domains_to_train:
        print(f"Evaluating {names[domain_id]}...")
        
        # Generate real samples
        real_samples = generators[domain_id](n_samples)
        
        # Generate samples from model with domain conditioning
        generated_samples = rf.sample(n_samples, dims[domain_id], domain_id, steps=50, device=device)
        generated_samples = generated_samples.cpu()
        
        # Compute metrics
        metrics = compute_evaluation_metrics(real_samples, generated_samples)
        
        real_samples_list.append(real_samples)
        generated_samples_list.append(generated_samples)
        eval_titles.append(names[domain_id])
        metrics_list.append(metrics)
        
        # Aggregate metrics
        for key in all_metrics.keys():
            all_metrics[key].append(metrics[key])
        
        # Log per-domain metrics to wandb
        if log_wandb:
            domain_log = {}
            for metric_name, value in metrics.items():
                domain_log[f'eval/{names[domain_id].replace(" ", "_")}/{metric_name}'] = value
            wandb.log(domain_log)
        
        # Print metrics summary
        print(f"  {names[domain_id]} Metrics:")
        print(f"    Wasserstein Distance: {metrics['wasserstein_distance']:.4f}")
        print(f"    MMD: {metrics['mmd']:.4f}")
        print(f"    Coverage: {metrics['coverage']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Mean Difference: {metrics['mean_difference']:.4f}")
        print(f"    Std Difference: {metrics['std_difference']:.4f}")
        print()
    
    # Compute and log aggregate metrics
    if log_wandb:
        aggregate_metrics = {}
        for metric_name, values in all_metrics.items():
            aggregate_metrics[f'eval/aggregate/mean_{metric_name}'] = np.mean(values)
            aggregate_metrics[f'eval/aggregate/std_{metric_name}'] = np.std(values)
            aggregate_metrics[f'eval/aggregate/min_{metric_name}'] = np.min(values)
            aggregate_metrics[f'eval/aggregate/max_{metric_name}'] = np.max(values)
        
        # Overall quality score (lower is better for most metrics)
        overall_scores = []
        for metrics in metrics_list:
            score = (metrics['wasserstein_distance'] + metrics['mmd'] + 
                    metrics['mean_difference'] + metrics['std_difference'] +
                    (1 - metrics['coverage']) + (1 - metrics['precision']))
            overall_scores.append(score)
        
        aggregate_metrics['eval/overall_quality_score'] = np.mean(overall_scores)
        aggregate_metrics['eval/overall_quality_std'] = np.std(overall_scores)
        
        wandb.log(aggregate_metrics)
    
    return real_samples_list, generated_samples_list, eval_titles, metrics_list

def main():
    parser = argparse.ArgumentParser(description='Multi-Domain Rectified Flow Training')
    parser.add_argument('--domains', nargs='+', type=int, default=[0, 1, 2], 
                      help='Domains to train on: 0=Double Gaussian, 1=Four Corners, 2=Spiral (default: all)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of MLP')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in MLP')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Custom wandb run name')
    
    args = parser.parse_args()
    
    # Validate domain indices
    valid_domains = [d for d in args.domains if d in [0, 1, 2]]
    if not valid_domains:
        print("No valid domains specified. Using all domains.")
        valid_domains = [0, 1, 2]
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Setup wandb logging
    use_wandb = not args.no_wandb
    
    # Train model
    print("="*50)
    print("MULTI-DOMAIN RECTIFIED FLOW TRAINING")
    print("="*50)
    
    rf, losses, names, dims, generators, domain_losses = train_multi_domain_rf(
        domains_to_train=valid_domains,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        log_wandb=use_wandb
    )
    
    # Plot training loss
    plt.figure(figsize=(12, 8))
    
    # Plot overall loss
    plt.subplot(2, 1, 1)
    plt.plot(losses, label='Total Loss', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot per-domain losses
    plt.subplot(2, 1, 2)
    for domain_id in valid_domains:
        plt.plot(domain_losses[domain_id], label=f'{names[domain_id]}', linewidth=2)
    plt.title('Per-Domain Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if args.save_plots:
        plt.savefig(output_dir / f'training_loss_domains_{"_".join(map(str, valid_domains))}.png')
    plt.show()
    
    # Log training curves to wandb
    if use_wandb:
        # Create training loss plot as wandb image
        wandb.log({"training_curves": wandb.Image(plt)})
    
    # Evaluate model
    print("\nEvaluating model...")
    real_samples_list, generated_samples_list, eval_titles, metrics_list = evaluate_model(
        rf, names, dims, generators, valid_domains, log_wandb=use_wandb
    )
    
    # Create summary metrics table
    print("\n" + "="*70)
    print("EVALUATION METRICS SUMMARY")
    print("="*70)
    print(f"{'Domain':<20} {'Wasserstein':<12} {'MMD':<8} {'Coverage':<10} {'Precision':<10}")
    print("-"*70)
    for title, metrics in zip(eval_titles, metrics_list):
        print(f"{title:<20} {metrics['wasserstein_distance']:<12.4f} {metrics['mmd']:<8.4f} "
              f"{metrics['coverage']:<10.4f} {metrics['precision']:<10.4f}")
    
    # Compute overall score (lower is better for most metrics)
    overall_scores = []
    for metrics in metrics_list:
        # Combine metrics (lower is better, except coverage and precision where higher is better)
        score = (metrics['wasserstein_distance'] + metrics['mmd'] + 
                metrics['mean_difference'] + metrics['std_difference'] +
                (1 - metrics['coverage']) + (1 - metrics['precision']))
        overall_scores.append(score)
    
    avg_score = np.mean(overall_scores)
    print(f"\nOverall Quality Score: {avg_score:.4f} (lower is better)")
    print("="*70)
    
    # Plot results
    save_path = None
    if args.save_plots:
        save_path = output_dir / f'results_domains_{"_".join(map(str, valid_domains))}.png'
    
    plot_distributions(real_samples_list, generated_samples_list, eval_titles, metrics_list, save_path)
    
    # Log final visualization to wandb
    if use_wandb:
        wandb.log({"final_results": wandb.Image(plt)})
        
        # Log summary table as wandb Table
        table_data = []
        for title, metrics in zip(eval_titles, metrics_list):
            table_data.append([
                title,
                metrics['wasserstein_distance'],
                metrics['mmd'],
                metrics['coverage'], 
                metrics['precision'],
                metrics['mean_difference'],
                metrics['std_difference']
            ])
        
        table = wandb.Table(
            columns=['Domain', 'Wasserstein', 'MMD', 'Coverage', 'Precision', 'Mean_Diff', 'Std_Diff'],
            data=table_data
        )
        wandb.log({"metrics_summary": table})
        
        # Log hyperparameters vs performance
        wandb.log({
            "final_summary/overall_quality_score": avg_score,
            "final_summary/num_domains_trained": len(valid_domains),
            "final_summary/final_loss": losses[-1] if losses else 0
        })
        
        wandb.finish()
    
    print(f"\nTraining completed! Trained on {len(valid_domains)} domains.")
    if args.save_plots:
        print(f"Results saved to {output_dir}")
    if use_wandb:
        print("Results logged to wandb project: toy_multi_dist_flower")

if __name__ == "__main__":
    main()