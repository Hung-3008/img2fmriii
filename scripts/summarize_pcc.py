import os
import pandas as pd

def main():
    subjects = [1, 2, 5, 7]
    base_dir = "exps/rfr"
    
    print("=== SUMMARY OF BEST PCC FOR EACH SUBJECT ===")
    
    rows_best = []
    for sub in subjects:
        csv_path = os.path.join(base_dir, f"rfr_dino4_gabor_sub{sub}", "history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Find the best row based on val_voxel_r
            if "val_voxel_r" in df.columns:
                best_voxel_idx = df["val_voxel_r"].idxmax()
                best_voxel_row = df.loc[best_voxel_idx]
                
                # Find the best row based on val_profile_r
                best_profile_idx = df["val_profile_r"].idxmax()
                best_profile_row = df.loc[best_profile_idx]
                
                rows_best.append({
                    "subject": f"sub{sub}",
                    "best_voxel_epoch": int(best_voxel_row["epoch"]),
                    "best_voxel_r": float(best_voxel_row["val_voxel_r"]),
                    "best_profile_epoch": int(best_profile_row["epoch"]),
                    "best_profile_r": float(best_profile_row["val_profile_r"])
                })
            else:
                print(f"Warning: val_voxel_r not found in {csv_path}")
        else:
            print(f"Warning: {csv_path} does not exist.")
            
    df_best = pd.DataFrame(rows_best)
    
    # Calculate means
    mean_voxel = df_best["best_voxel_r"].mean()
    mean_profile = df_best["best_profile_r"].mean()
    
    # Add mean row to the dataframe
    mean_row = {
        "subject": "mean",
        "best_voxel_epoch": None,
        "best_voxel_r": mean_voxel,
        "best_profile_epoch": None,
        "best_profile_r": mean_profile
    }
    df_best_with_mean = pd.concat([df_best, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save to CSV
    output_csv_path = os.path.join(base_dir, "best_pcc_summary.csv")
    df_best_with_mean.to_csv(output_csv_path, index=False)
    print(f"\nSaved best PCC summary to: {output_csv_path}")
    print(df_best_with_mean.to_string(index=False))
    
    # Let's also compile the top 4 best PCC (voxel_r) for each subject just in case
    print("\n=== TOP 4 BEST PCC (voxel_r) FOR EACH SUBJECT ===")
    top4_rows = []
    for sub in subjects:
        csv_path = os.path.join(base_dir, f"rfr_dino4_gabor_sub{sub}", "history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "val_voxel_r" in df.columns:
                # Get top 4 sorted by val_voxel_r descending
                df_sorted = df.sort_values(by="val_voxel_r", ascending=False).head(4)
                for idx, row in df_sorted.iterrows():
                    top4_rows.append({
                        "subject": f"sub{sub}",
                        "epoch": int(row["epoch"]),
                        "step": int(row["step"]),
                        "val_voxel_r": float(row["val_voxel_r"]),
                        "val_profile_r": float(row["val_profile_r"]),
                        "train_loss": float(row["train_loss"])
                    })
    
    df_top4 = pd.DataFrame(top4_rows)
    top4_csv_path = os.path.join(base_dir, "top4_pcc_each_subject.csv")
    df_top4.to_csv(top4_csv_path, index=False)
    print(f"Saved top 4 PCC each subject to: {top4_csv_path}")
    
    # Group by subject and show mean of top 4 for each subject
    print("\nMean of top 4 PCC (voxel_r) per subject:")
    print(df_top4.groupby("subject")["val_voxel_r"].mean())

if __name__ == "__main__":
    main()
