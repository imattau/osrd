"""Compatibility entry point for the memory-bounded RIDE GNN trainer."""

from osrd_ops_intelligence.training.train_ride_gnn_streaming import main, train

__all__ = ["main", "train"]


if __name__ == "__main__":
    main()
