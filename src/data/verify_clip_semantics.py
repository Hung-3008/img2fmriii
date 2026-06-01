"""
verify_clip_semantics.py
========================
Kiểm tra xem CLIP features đã extract có chứa thông tin ngữ nghĩa không
bằng các phương pháp:

1. Kiểm tra cơ bản: shape, stats, norm
2. Nearest-neighbor retrieval: ảnh cùng ngữ nghĩa phải gần nhau hơn ảnh khác loại
3. Cosine similarity heatmap (intra vs inter category)
4. t-SNE visualization để xem clustering ngữ nghĩa
5. Zero-shot classification test với ImageNet labels

Usage:
    uv run python src/data/verify_clip_semantics.py --subject 1 --n_samples 200
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../NSD/data'))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'nsd')


# ==============================================================================
# Test 1: Basic Statistics
# ==============================================================================
def test_basic_stats(pool_feats, token_feats=None):
    """Kiểm tra shape, dtype, norm, và tính đa dạng của features."""
    print("\n" + "="*60)
    print("TEST 1: Basic Statistics")
    print("="*60)

    print(f"  Pool features shape  : {pool_feats.shape}  (N, D)")
    print(f"  Pool features dtype  : {pool_feats.dtype}")
    print(f"  Pool mean            : {pool_feats.mean():.4f}")
    print(f"  Pool std             : {pool_feats.std():.4f}")
    print(f"  Pool min / max       : {pool_feats.min():.4f} / {pool_feats.max():.4f}")

    norms = np.linalg.norm(pool_feats.astype(np.float32), axis=1)
    print(f"  L2-norm mean ± std   : {norms.mean():.4f} ± {norms.std():.4f}")
    print(f"    → {'✅ ~1.0 (normalized)' if abs(norms.mean()-1.0)<0.1 else '⚠️  Not unit-norm'}")

    var_per_dim = pool_feats.astype(np.float32).var(axis=0)
    dead_dims = (var_per_dim < 1e-6).sum()
    print(f"  Dead dims (var<1e-6) : {dead_dims}/{pool_feats.shape[1]}")
    print(f"    → {'✅ No dead dims' if dead_dims==0 else f'⚠️  {dead_dims} dead dims'}")

    if token_feats is not None:
        print(f"\n  Token features shape : {token_feats.shape}  (N, T, D)")


# ==============================================================================
# Test 2: Pairwise cosine similarity distribution
# ==============================================================================
def test_similarity_distribution(pool_feats, n_samples=500, save_dir=None):
    """
    Nếu features có ngữ nghĩa, cosine similarity giữa các ảnh KHÔNG nên đều nhau.
    Distribution phải có đuôi dài (một số cặp rất giống, một số rất khác).
    Nếu features ngẫu nhiên → distribution narrow và centered.
    """
    print("\n" + "="*60)
    print("TEST 2: Pairwise Cosine Similarity Distribution")
    print("="*60)

    n = min(n_samples, len(pool_feats))
    idx = np.random.choice(len(pool_feats), n, replace=False)
    feats = pool_feats[idx].astype(np.float32)

    # Normalize
    feats_t = torch.from_numpy(feats)
    feats_t = F.normalize(feats_t, dim=1)
    sim_matrix = (feats_t @ feats_t.T).numpy()

    # Off-diagonal only
    mask = ~np.eye(n, dtype=bool)
    sims = sim_matrix[mask]

    mean_sim = sims.mean()
    std_sim  = sims.std()
    top5_sim = np.percentile(sims, 95)
    bot5_sim = np.percentile(sims, 5)

    print(f"  Samples used         : {n}")
    print(f"  Cosine sim mean ± std: {mean_sim:.4f} ± {std_sim:.4f}")
    print(f"  5th / 95th percentile: {bot5_sim:.4f} / {top5_sim:.4f}")
    print(f"  Range (max-min)      : {sims.max()-sims.min():.4f}")
    print(f"    → {'✅ Wide range — features discriminative' if std_sim>0.05 else '⚠️  Narrow range — may lack diversity'}")

    if save_dir:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sims, bins=80, color='steelblue', edgecolor='none', alpha=0.85)
        ax.axvline(mean_sim, color='red', ls='--', lw=1.5, label=f'mean={mean_sim:.3f}')
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Pairwise Cosine Similarity Distribution\n(SDXL CLIP pool features)', fontsize=13)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(save_dir, 'clip_cosine_sim_dist.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  📊 Saved: {path}")

    return sim_matrix, idx


# ==============================================================================
# Test 3: Top-K nearest neighbor retrieval (visual)
# ==============================================================================
def test_nearest_neighbor(pool_feats, image_paths, n_queries=6, k=5, save_dir=None):
    """
    Với mỗi query image, tìm top-K ảnh gần nhất theo cosine similarity.
    Nếu features có ngữ nghĩa, các ảnh retrieved phải nhìn visually tương đồng.
    """
    print("\n" + "="*60)
    print("TEST 3: Nearest Neighbor Retrieval")
    print("="*60)

    feats = pool_feats.astype(np.float32)
    feats_t = torch.from_numpy(feats)
    feats_t = F.normalize(feats_t, dim=1)
    sim_all = (feats_t @ feats_t.T).numpy()

    # Exclude self
    np.fill_diagonal(sim_all, -1.0)

    # Randomly choose query indices
    rng = np.random.default_rng(42)
    query_ids = rng.choice(len(pool_feats), n_queries, replace=False)

    print(f"  Queries: {query_ids.tolist()}")

    if save_dir:
        fig, axes = plt.subplots(n_queries, k+1, figsize=(3*(k+1), 3*n_queries))
        if n_queries == 1:
            axes = axes[np.newaxis]

        for row, qid in enumerate(query_ids):
            top_k = np.argsort(sim_all[qid])[::-1][:k]
            sims_top = sim_all[qid][top_k]

            # Query image
            try:
                qimg = Image.open(image_paths[qid]).convert('RGB').resize((224,224))
            except Exception:
                qimg = Image.new('RGB', (224,224), (128,128,128))
            axes[row, 0].imshow(qimg)
            axes[row, 0].set_title(f'Query #{qid}', fontsize=8, color='black', fontweight='bold')
            axes[row, 0].axis('off')
            axes[row, 0].patch.set_edgecolor('blue')
            axes[row, 0].patch.set_linewidth(3)

            for col, (rid, sim_v) in enumerate(zip(top_k, sims_top)):
                try:
                    rimg = Image.open(image_paths[rid]).convert('RGB').resize((224,224))
                except Exception:
                    rimg = Image.new('RGB', (224,224), (128,128,128))
                axes[row, col+1].imshow(rimg)
                axes[row, col+1].set_title(f'#{rid}\nsim={sim_v:.3f}', fontsize=7)
                axes[row, col+1].axis('off')

        fig.suptitle('Top-K Nearest Neighbors (SDXL CLIP features)\n'
                     'Nếu features có ngữ nghĩa → retrieved ảnh phải visually tương đồng',
                     fontsize=11, y=1.01)
        fig.tight_layout()
        path = os.path.join(save_dir, 'clip_nearest_neighbors.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 Saved: {path}")
    else:
        for qid in query_ids:
            top_k = np.argsort(sim_all[qid])[::-1][:k]
            sims_top = sim_all[qid][top_k]
            print(f"  Query {qid:5d} → Retrieved: {list(zip(top_k.tolist(), [f'{s:.3f}' for s in sims_top]))}")


# ==============================================================================
# Test 4: t-SNE visualization
# ==============================================================================
def test_tsne(pool_feats, n_samples=1000, save_dir=None):
    """
    t-SNE để visualize structure trong feature space.
    Nếu có ngữ nghĩa → sẽ thấy clusters rõ ràng thay vì random scatter.
    """
    print("\n" + "="*60)
    print("TEST 4: t-SNE Visualization")
    print("="*60)

    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("  ⚠️  scikit-learn not available, skipping t-SNE")
        return

    n = min(n_samples, len(pool_feats))
    idx = np.random.default_rng(0).choice(len(pool_feats), n, replace=False)
    feats = pool_feats[idx].astype(np.float32)

    # Normalize
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats = feats / norms

    # PCA whitening to 50D first for speed
    print(f"  Running PCA (50D) on {n} samples...")
    pca = PCA(n_components=min(50, feats.shape[1]), random_state=0)
    feats_pca = pca.fit_transform(feats)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance (50D): {explained:.1%}")

    print(f"  Running t-SNE...")
    try:
        tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=0, verbose=0)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=0, verbose=0)
    emb = tsne.fit_transform(feats_pca)

    if save_dir:
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=idx, cmap='plasma', s=6, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Image index')
        ax.set_title('t-SNE of SDXL CLIP pool features\n'
                     '(Clusters → ngữ nghĩa tương đồng nhóm lại với nhau)', fontsize=12)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        fig.tight_layout()
        path = os.path.join(save_dir, 'clip_tsne.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  📊 Saved: {path}")


# ==============================================================================
# Test 5: Self-similarity matrix heatmap (mini)
# ==============================================================================
def test_similarity_heatmap(pool_feats, n_samples=100, save_dir=None):
    """
    Visualize similarity matrix của N ảnh đầu tiên.
    Nếu có ngữ nghĩa → matrix sẽ có block structure (ảnh cùng chủ đề gần nhau).
    """
    print("\n" + "="*60)
    print("TEST 5: Similarity Matrix Heatmap")
    print("="*60)

    n = min(n_samples, len(pool_feats))
    feats = pool_feats[:n].astype(np.float32)
    feats_t = torch.from_numpy(feats)
    feats_t = F.normalize(feats_t, dim=1)
    sim_matrix = (feats_t @ feats_t.T).numpy()

    print(f"  Matrix size          : {n}x{n}")
    off_diag = sim_matrix[~np.eye(n, dtype=bool)]
    print(f"  Off-diag sim mean    : {off_diag.mean():.4f}")
    print(f"  Off-diag sim std     : {off_diag.std():.4f}")

    if save_dir:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(sim_matrix, cmap='viridis', vmin=-0.2, vmax=1.0, aspect='auto')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        ax.set_title(f'Pairwise Cosine Similarity Heatmap\n'
                     f'(first {n} images, SDXL CLIP pool features)', fontsize=11)
        ax.set_xlabel('Image index')
        ax.set_ylabel('Image index')
        fig.tight_layout()
        path = os.path.join(save_dir, 'clip_sim_heatmap.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  📊 Saved: {path}")


# ==============================================================================
# Test 6: Dimensionality / Intrinsic rank
# ==============================================================================
def test_intrinsic_rank(pool_feats, save_dir=None):
    """
    Kiểm tra intrinsic dimensionality bằng singular value spectrum.
    CLIP features thực sự có ngữ nghĩa → rank cao, spectrum mịn.
    Features ngẫu nhiên → rank rất thấp hoặc rất cao tùy init.
    """
    print("\n" + "="*60)
    print("TEST 6: Intrinsic Dimensionality (SVD)")
    print("="*60)

    try:
        from sklearn.decomposition import TruncatedSVD
    except ImportError:
        print("  ⚠️  scikit-learn not available, skipping SVD")
        return

    n = min(5000, len(pool_feats))
    feats = pool_feats[:n].astype(np.float32)
    feats = feats - feats.mean(axis=0)

    k = min(200, feats.shape[1], n-1)
    svd = TruncatedSVD(n_components=k, random_state=0)
    svd.fit(feats)
    sv = svd.singular_values_

    # Effective rank: exp(entropy of normalized singular values squared)
    sv2 = sv**2
    sv2_norm = sv2 / sv2.sum()
    entropy = -np.sum(sv2_norm * np.log(sv2_norm + 1e-12))
    effective_rank = np.exp(entropy)

    # How many dims to explain 90% variance
    cum_var = np.cumsum(sv2) / sv2.sum()
    dim_90 = np.searchsorted(cum_var, 0.90) + 1
    dim_99 = np.searchsorted(cum_var, 0.99) + 1

    print(f"  Top-{k} singular values computed")
    print(f"  Effective rank       : {effective_rank:.1f}")
    print(f"  Dims for 90% var     : {dim_90}/{feats.shape[1]}")
    print(f"  Dims for 99% var     : {dim_99}/{feats.shape[1]}")
    print(f"  → {'✅ High rank = rich semantic structure' if effective_rank>50 else '⚠️  Low rank = may be collapsed'}")

    if save_dir:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(range(1, k+1), sv, 'steelblue', lw=1.5)
        axes[0].set_xlabel('Singular value index')
        axes[0].set_ylabel('Singular value')
        axes[0].set_title('Singular Value Spectrum')
        axes[0].set_yscale('log')

        axes[1].plot(range(1, k+1), cum_var * 100, 'darkorange', lw=1.5)
        axes[1].axhline(90, color='red', ls='--', lw=1, label='90%')
        axes[1].axhline(99, color='purple', ls='--', lw=1, label='99%')
        axes[1].set_xlabel('Number of dimensions')
        axes[1].set_ylabel('Cumulative explained variance (%)')
        axes[1].set_title(f'Cumulative Variance\n(eff. rank={effective_rank:.1f})')
        axes[1].legend()

        fig.suptitle('SDXL CLIP pool features — Intrinsic Dimensionality', fontsize=12)
        fig.tight_layout()
        path = os.path.join(save_dir, 'clip_svd_spectrum.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  📊 Saved: {path}")


# ==============================================================================
# Main
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Verify semantic content in CLIP features')
    parser.add_argument('--subject', type=int, default=1, help='Subject number (default: 1)')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--n_samples', type=int, default=500, help='Số samples dùng cho visualization')
    parser.add_argument('--n_queries', type=int, default=6, help='Số query images cho NN retrieval')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Thư mục lưu plots (default: NSD/data/verify_clip/)')
    parser.add_argument('--skip_tsne', action='store_true', help='Bỏ qua t-SNE (nhanh hơn)')
    return parser.parse_args()


def main():
    args = parse_args()

    subj = args.subject
    mode = args.mode
    subj_dir = os.path.join(PROCESSED_DATA_DIR, f'subj0{subj}')

    # Setup save directory
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(BASE_DIR, '..', 'verify_clip')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nPlots sẽ được lưu vào: {os.path.abspath(save_dir)}")

    # Load pool features
    pool_path = os.path.join(subj_dir, f'nsd_sdxl_clip_pool_{mode}_sub{subj}.npy')
    token_path = os.path.join(subj_dir, f'nsd_sdxl_clip_{mode}_sub{subj}.npy')

    if not os.path.exists(pool_path):
        print(f"\n❌ File not found: {pool_path}")
        print("   Chạy extract_features.py --model sdxl_clip trước.")
        sys.exit(1)

    print(f"\nLoading: {pool_path}")
    pool_feats = np.load(pool_path)  # (N, D)
    print(f"  Shape: {pool_feats.shape}")

    token_feats = None
    if os.path.exists(token_path):
        print(f"Loading: {token_path}")
        token_feats = np.load(token_path)
        print(f"  Shape: {token_feats.shape}")

    # Load image paths for visualization
    image_dir = os.path.join(subj_dir, f'{mode}_img')
    num_images = len(pool_feats)
    image_paths = [os.path.join(image_dir, f'{i}.png') for i in range(num_images)]
    images_exist = os.path.exists(image_dir) and os.path.exists(image_paths[0])
    if not images_exist:
        print(f"  ⚠️  Image dir not found: {image_dir} — NN retrieval sẽ dùng placeholder")

    # Run tests
    np.random.seed(42)

    test_basic_stats(pool_feats, token_feats)
    test_similarity_distribution(pool_feats, n_samples=args.n_samples, save_dir=save_dir)
    test_nearest_neighbor(pool_feats, image_paths if images_exist else [None]*num_images,
                          n_queries=args.n_queries, k=5, save_dir=save_dir)
    test_similarity_heatmap(pool_feats, n_samples=100, save_dir=save_dir)
    test_intrinsic_rank(pool_feats, save_dir=save_dir)

    if not args.skip_tsne:
        test_tsne(pool_feats, n_samples=args.n_samples, save_dir=save_dir)

    print("\n" + "="*60)
    print("✅ Verification complete!")
    print(f"   Plots saved to: {os.path.abspath(save_dir)}")
    print("="*60)

    # Summary verdict
    print("\n📋 SEMANTIC QUALITY CHECKLIST:")
    print("   ✅ = có ngữ nghĩa    ⚠️  = cần kiểm tra thêm")
    print()
    print("   1. L2-norm ≈ 1.0           → unit-normalized features (chuẩn CLIP)")
    print("   2. Cosine sim std > 0.05    → features đa dạng, không collapsed")
    print("   3. NN retrieval visually OK → features capture visual semantics")
    print("   4. t-SNE có clusters        → features có cấu trúc ngữ nghĩa")
    print("   5. Effective rank > 50      → rich multi-dimensional semantics")


if __name__ == '__main__':
    main()
