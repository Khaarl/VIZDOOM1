import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import argparse
import time
import logging
from datetime import datetime

# Mount Google Drive
try:
    drive.mount('/content/drive')
except:
    print("Drive already mounted or not in Colab environment")

# Configuration - Updated paths for Google Drive
DRIVE_ROOT = "/content/drive/MyDrive/ViZDoom"  # Updated path
METRICS_DIR = os.path.join(DRIVE_ROOT, "metrics")
LOGS_DIR = os.path.join(DRIVE_ROOT, "logs")
MODELS_DIR = os.path.join(DRIVE_ROOT, "models")

# Ensure directories exist
for directory in [DRIVE_ROOT, METRICS_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

def setup_colab_logging(name):
    """Setup logging for Colab environment"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def get_latest_run():
    """Get the most recent run ID from metrics directory"""
    metrics_files = glob.glob(os.path.join(METRICS_DIR, "*.csv"))
    if not metrics_files:
        return None
    latest_file = max(metrics_files, key=os.path.getctime)
    return os.path.basename(latest_file).split('_')[0]

def load_and_combine_metrics(model_dir):
    """
    Loads and combines all CSV files from the specified directory.
    Handles duplicate column names effectively by keeping only the first occurrence.
    """
    print(f"Looking for CSV files in: {model_dir}")  # Debug print
    metrics_files = glob.glob(os.path.join(model_dir, "*.csv"))
    
    if not metrics_files:
        # Try alternative path format
        alt_path = model_dir.replace("/content/drive/MyDrive", "/content/drive/My Drive")
        print(f"Trying alternative path: {alt_path}")  # Debug print
        metrics_files = glob.glob(os.path.join(alt_path, "*.csv"))
    
    if not metrics_files:
        print(f"No CSV files found in either {model_dir} or alternative path")
        return None

    all_metrics = []
    for file in metrics_files:
        try:
            df = pd.read_csv(file)
            print(f"Loaded file: {file}")
            print(f"Original columns: {df.columns.tolist()}")

            # --- Keep First Occurrence of Duplicate Columns ---
            # Rename columns, keeping track of duplicates
            new_cols = []
            seen_cols = {}
            for col in df.columns:
                col_name = col.replace("_x", "").replace("_y", "")
                if col_name in seen_cols:
                    seen_cols[col_name] += 1
                else:
                    seen_cols[col_name] = 0
                
                if seen_cols[col_name] > 0:
                  print(f"Duplicate column name after cleaning: {col_name} in file {file}")

                new_cols.append(col_name)

            df.columns = new_cols
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

            print(f"Cleaned columns: {df.columns.tolist()}")

            df['source_file'] = os.path.basename(file)
            all_metrics.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    if not all_metrics:
        print("No valid CSV files could be loaded.")
        return None

    # --- Concatenate and Handle Empty/NA Columns ---
    valid_metrics = [df for df in all_metrics if not df.empty and not df.isna().all().all()]
    if not valid_metrics:
        print("No valid data found in CSV files for concatenation.")
        return None
    combined_metrics = pd.concat(valid_metrics, ignore_index=True)

    # --- Handle missing 'episode' column ---
    if 'episode' not in combined_metrics.columns:
        episode_counter = 0
        for df in valid_metrics:
            if 'episode' not in df.columns:
                df['episode'] = range(1, len(df) + 1)
                df['episode'] += episode_counter
            episode_counter = df['episode'].max()
        combined_metrics = pd.concat(valid_metrics, ignore_index=True)

    print(f"Combined DataFrame columns: {combined_metrics.columns.tolist()}")

    return combined_metrics

# ... (The visualize_metrics function remains the same) ...
def visualize_metrics(metrics_df, output_dir=None, clear_previous=False):
    """Visualizes metrics, dynamically adapting to available columns."""
    if metrics_df is None or metrics_df.empty:
        print("No data to visualize.")
        return

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def save_or_show_plot(fig, title, output_dir):
        if output_dir:
            filepath = os.path.join(output_dir, f"{title}.png")
            fig.savefig(filepath)
            print(f"Saved plot: {filepath}")
        else:
            plt.show()
        plt.close(fig)

    # --- Get available numeric columns (excluding 'episode' and 'source_file') ---
    available_numeric_cols = metrics_df.select_dtypes(include='number').columns.tolist()
    available_numeric_cols = [col for col in available_numeric_cols if col != 'episode' and col != 'source_file']

    # --- Define potential plots and their required columns ---
    potential_plots = {
        "reward_per_episode": {"x": "episode", "y": "reward"},
        "steps_per_episode": {"x": "episode", "y": "steps"},
        "epsilon_decay": {"x": "episode", "y": "epsilon"},
        "survival_time_per_episode": {"x": "episode", "y": "survival_time"},
        "damage_taken_vs_inflicted": {"x": "episode", "y": ["damage_taken", "damage_inflicted"]},
        "avg_q_value_per_episode": {"x": "episode", "y": "avg_q_value"},
        "loss_per_episode": {"x": "episode", "y": "loss"},
        "action_diversity_per_episode": {"x": "episode", "y": "action_diversity"},
        "ram_gpu_usage_per_episode": {"x": "episode", "y": ["ram_usage", "gpu_memory_usage"]},
        "training_time_per_episode": {"x": "episode", "y": "training_time"},
        "avg_inference_time_per_episode": {"x": "episode", "y": "avg_inference_time"},
    }

    # --- Generate plots based on available columns ---
    for plot_name, required_cols in potential_plots.items():
        try:
            if all(col in available_numeric_cols for col in required_cols.get("y", [])):
                fig, ax = plt.subplots(figsize=(12, 6))
                y_data = required_cols["y"]

                # Handle single vs. multiple y columns
                if isinstance(y_data, list):
                    for y_col in y_data:
                        sns.lineplot(data=metrics_df, x=required_cols["x"], y=y_col, label=y_col, ax=ax)
                    ax.legend()
                else:
                    sns.lineplot(data=metrics_df, x=required_cols["x"], y=y_data, ax=ax)

                ax.set_title(plot_name.replace("_", " ").title())
                ax.set_xlabel(required_cols["x"].title())
                ax.set_ylabel(y_data.title() if isinstance(y_data, str) else "Value")

                save_or_show_plot(fig, plot_name, output_dir)
            else:
                print(f"Skipping {plot_name} plot: Required columns not found.")
        except Exception as e:
            print(f"Error generating {plot_name} plot: {e}")

    # --- Reward Comparison Across Different Runs (Optional) ---
    if 'source_file' in metrics_df.columns and 'reward' in available_numeric_cols:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=metrics_df, x='source_file', y='reward', ax=ax)
            ax.set_title('Comparison of Rewards Across Different Runs')
            ax.set_xlabel('Training Run')
            ax.set_ylabel('Reward')
            plt.xticks(rotation=45, ha='right')
            save_or_show_plot(fig, "reward_comparison_across_runs", output_dir)
        except Exception as e:
            print(f"Error generating Reward Comparison Across Runs plot: {e}")

    # --- Correlation Heatmap (Numeric Columns Only) ---
    try:
        numeric_df = metrics_df[available_numeric_cols]
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap of Metrics')
        save_or_show_plot(fig, "correlation_heatmap", output_dir)
    except Exception as e:
        print(f"Error generating Correlation Heatmap: {e}")

# Add real-time monitoring capabilities
def monitor_training(run_id=None):
    """Monitor training progress in real-time"""
    logger = setup_colab_logging("dashboard")
    
    if run_id is None:
        run_id = get_latest_run()
        if run_id is None:
            logger.error("No metrics files found")
            return
    
    metrics_file = os.path.join(METRICS_DIR, f"{run_id}_metrics.csv")
    
    logger.info(f"Monitoring metrics file: {metrics_file}")
    while True:
        try:
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                # Update plots in real-time
                visualize_metrics(df, clear_previous=True)
                plt.pause(10)  # Update every 10 seconds
            else:
                logger.warning(f"Waiting for metrics file: {metrics_file}")
                time.sleep(10)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error monitoring metrics: {e}")
            time.sleep(10)

# Remove argparse for Colab compatibility
def run_dashboard(mode='analyze', run_id=None):
    """Main dashboard function that can be called from Colab"""
    print(f"Dashboard running in {mode} mode")  # Debug print
    print(f"Looking for metrics in: {METRICS_DIR}")  # Debug print
    
    if mode == 'monitor':
        monitor_training(run_id)
    else:
        metrics_df = load_and_combine_metrics(METRICS_DIR)
        if metrics_df is not None:
            output_directory = os.path.join(METRICS_DIR, "plots")
            visualize_metrics(metrics_df, output_directory)
        else:
            print("No metrics data found to visualize")

# Replace main() with Colab-friendly version
if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        # Default to analysis mode in Colab
        run_dashboard(mode='analyze')
    else:
        # For command line usage (non-Colab)
        try:
            import argparse
            parser = argparse.ArgumentParser(description='ViZDoom Training Dashboard')
            parser.add_argument('--run_id', type=str, help='Specific run ID to monitor')
            parser.add_argument('--mode', choices=['monitor', 'analyze'], 
                              default='analyze',
                              help='Monitor training in real-time or analyze completed runs')
            # Only parse known args to avoid conflicts with Jupyter/Colab
            args, unknown = parser.parse_known_args()
            run_dashboard(args.mode, args.run_id)
        except Exception as e:
            print(f"Command line parsing failed: {e}")
            # Fall back to default analysis mode
            run_dashboard(mode='analyze')

# Add Colab usage examples as comments
"""
# Usage in Colab:

# For analysis mode:
run_dashboard(mode='analyze')

# For monitoring mode:
run_dashboard(mode='monitor')

# For monitoring specific run:
run_dashboard(mode='monitor', run_id='YourRunID')
"""