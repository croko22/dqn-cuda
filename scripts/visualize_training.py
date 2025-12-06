import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_learning_curves(episodes_df, window=10):
    """Plot reward and loss learning curves with moving averages."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    if len(episodes_df) > 0:
        ax.plot(episodes_df['episode'], episodes_df['reward'], alpha=0.3, label='Raw Reward')
    else:
        ax.text(0.5, 0.5, 'No data yet', ha='center', va='center')
    
    # Moving average
    if len(episodes_df) >= window:
        episodes_df['reward_ma'] = episodes_df['reward'].rolling(window=window, min_periods=1).mean()
        ax.plot(episodes_df['episode'], episodes_df['reward_ma'], 
                linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Training Loss
    ax = axes[0, 1]
    ax.plot(episodes_df['episode'], episodes_df['avg_loss'], alpha=0.6, label='Avg Loss')
    
    # Moving average for loss
    if len(episodes_df) >= window:
        episodes_df['loss_ma'] = episodes_df['avg_loss'].rolling(window=window, min_periods=1).mean()
        ax.plot(episodes_df['episode'], episodes_df['loss_ma'], 
                linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Episode Length (Steps)
    ax = axes[1, 0]
    ax.plot(episodes_df['episode'], episodes_df['steps'], alpha=0.6, label='Steps per Episode')
    
    # Moving average for steps
    if len(episodes_df) >= window:
        episodes_df['steps_ma'] = episodes_df['steps'].rolling(window=window, min_periods=1).mean()
        ax.plot(episodes_df['episode'], episodes_df['steps_ma'], 
                linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Epsilon Decay
    ax = axes[1, 1]
    ax.plot(episodes_df['episode'], episodes_df['epsilon'], linewidth=2, color='purple')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon) Decay')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_step_analysis(steps_df):
    """Plot detailed step-level analysis (if available)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Q-values over time
    ax = axes[0, 0]
    ax.plot(steps_df['step'], steps_df['q_value'], alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Max Q-Value')
    ax.set_title('Q-Value Evolution')
    ax.grid(True, alpha=0.3)
    
    # 2. Loss over steps
    ax = axes[0, 1]
    ax.plot(steps_df['step'], steps_df['loss'], alpha=0.5, color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Step-by-Step Loss')
    ax.grid(True, alpha=0.3)
    
    # 3. Action distribution
    ax = axes[1, 0]
    action_counts = steps_df['action'].value_counts().sort_index()
    ax.bar(action_counts.index, action_counts.values)
    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Reward distribution
    ax = axes[1, 1]
    ax.hist(steps_df['reward'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Step Reward Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def print_statistics(episodes_df):
    """Print training statistics."""
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    
    print(f"\nTotal Episodes: {len(episodes_df)}")
    print(f"Total Steps: {episodes_df['steps'].sum()}")
    
    if len(episodes_df) == 0:
        print("\nNo episodes logged yet.")
        print("="*60 + "\n")
        return
    
    print("\nREWARD STATISTICS:")
    print(f"  Mean Reward: {episodes_df['reward'].mean():.4f}")
    print(f"  Std Reward: {episodes_df['reward'].std():.4f}")
    print(f"  Max Reward: {episodes_df['reward'].max():.4f}")
    print(f"  Min Reward: {episodes_df['reward'].min():.4f}")
    
    # Last 10% of episodes
    last_10_percent = int(len(episodes_df) * 0.1)
    if last_10_percent > 0:
        recent_reward = episodes_df['reward'].tail(last_10_percent).mean()
        print(f"  Recent Reward (last 10%): {recent_reward:.4f}")
    
    print("\nLOSS STATISTICS:")
    print(f"  Mean Loss: {episodes_df['avg_loss'].mean():.4f}")
    print(f"  Final Loss: {episodes_df['avg_loss'].iloc[-1]:.4f}")
    
    print("\nEPISODE LENGTH:")
    print(f"  Mean Steps: {episodes_df['steps'].mean():.2f}")
    print(f"  Max Steps: {episodes_df['steps'].max()}")
    print(f"  Min Steps: {episodes_df['steps'].min()}")
    
    print("\nEXPLORATION:")
    print(f"  Final Epsilon: {episodes_df['epsilon'].iloc[-1]:.4f}")
    
    print("="*60 + "\n")

def main():
    import glob

    if len(sys.argv) < 2:
        # Default to finding latest in results/
        search_path = "results/*_episodes.csv"
        files = glob.glob(search_path)
        if not files:
            search_path = "../results/*_episodes.csv"
            files = glob.glob(search_path)
        
        if files:
            # Sort by modification time
            latest_file = max(files, key=os.path.getmtime)
            print(f"No argument provided. Using latest log file: {latest_file}")
            base_filename = latest_file
        else:
            print("Usage: python visualize_training.py <log_basename_or_file>")
            print("Example: python visualize_training.py results/dqn_results")
            sys.exit(1)
    else:
        base_filename = sys.argv[1]

    # If argument is a directory, find latest in it
    if os.path.isdir(base_filename):
        search_path = os.path.join(base_filename, "*_episodes.csv")
        files = glob.glob(search_path)
        if files:
            latest_file = max(files, key=os.path.getmtime)
            print(f"Directory provided. Using latest log file: {latest_file}")
            base_filename = latest_file
        else:
            print(f"Error: No log files found in directory {base_filename}")
            sys.exit(1)
            
    # If argument is a prefix (e.g. results/dqn_results) but not a file, try to find latest match
    if not os.path.exists(base_filename) and not os.path.exists(f"{base_filename}_episodes.csv"):
        # Try as prefix
        search_path = f"{base_filename}*_episodes.csv"
        files = glob.glob(search_path)
        if files:
            latest_file = max(files, key=os.path.getmtime)
            print(f"Prefix provided. Using latest matching log file: {latest_file}")
            base_filename = latest_file
    
    # Strip extension if user provided full filename
    if base_filename.endswith("_episodes.csv"):
        base_filename = base_filename.replace("_episodes.csv", "")
    elif base_filename.endswith("_steps.csv"):
        base_filename = base_filename.replace("_steps.csv", "")
    elif base_filename.endswith(".csv"):
        base_filename = base_filename.replace(".csv", "")
    
    # Construct paths
    # Try to find the file in potential locations
    candidates = [
        base_filename,
        os.path.join("results", base_filename),
        os.path.join("..", "results", base_filename),
        os.path.join(os.path.dirname(__file__), "..", "results", base_filename)
    ]
    
    found_base = None
    for cand in candidates:
        if os.path.exists(f"{cand}_episodes.csv"):
            found_base = cand
            break
            
    if found_base:
        base_filename = found_base
        print(f"Found log files at: {base_filename}_episodes.csv")
    else:
        # Fallback to original behavior to show error
        pass

    episodes_file = f"{base_filename}_episodes.csv"
    steps_file = f"{base_filename}_steps.csv"
    
    # Check if files exist
    if not os.path.exists(episodes_file):
        print(f"Error: {episodes_file} not found!")
        print("Checked locations:")
        for cand in candidates:
            print(f"  - {cand}_episodes.csv")
        sys.exit(1)
    
    # Load episode data
    print(f"Loading {episodes_file}...")
    episodes_df = pd.read_csv(episodes_file)
    
    # Print statistics
    print_statistics(episodes_df)
    
    # Plot learning curves
    print("Generating learning curves...")
    fig1 = plot_learning_curves(episodes_df)
    output_file1 = f"{base_filename}_learning_curves.png"
    fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file1}")
    
    # Plot step analysis if available
    if os.path.exists(steps_file):
        print(f"\nLoading {steps_file}...")
        steps_df = pd.read_csv(steps_file)
        
        print("Generating step analysis...")
        fig2 = plot_step_analysis(steps_df)
        output_file2 = f"{base_filename}_step_analysis.png"
        fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file2}")
    else:
        print(f"\nNote: {steps_file} not found (step-level logging disabled)")
    
    # Show plots
    print("\nDisplaying plots...")
    plt.show()

if __name__ == "__main__":
    main()
