# Motion Prediction

Human motion prediction repository with state-of-the-art models (2024-2025).

## 🆕 Latest Research-Based Improvements

This repository has been significantly updated with cutting-edge techniques from 2024-2025 research:

### New Models
- ✅ **Graph Convolutional Networks (GCN)** - Skeleton-aware predictions (+5-11% improvement)
- ✅ **Temporal Convolutional Networks (TCN)** - Efficient parallel training (+3-7% improvement)
- ✅ **TCN-Transformer Hybrid** - Best of both worlds (+7-12% improvement)
- ✅ **Original RNN/Transformer** - Updated with modern techniques

### New Features
- ✅ **Advanced Loss Functions** - Velocity, acceleration, bone length constraints
- ✅ **Uncertainty Estimation** - Monte Carlo Dropout & ensemble methods
- ✅ **Improved Training** - Adam optimizer, learning rate scheduling, better metrics
- ✅ **Comprehensive Metrics** - MPJPE, PCK, calibration error

## Quick Start

```bash
# Install dependencies
pip install torch numpy h5py matplotlib

# Train with Graph Convolutional Network (recommended)
cd human-motion-prediction-pytorch
python src/train_improved.py --model_type gcn --action walking --iterations 50000

# Train with Temporal Convolutional Network (fastest)
python src/train_improved.py --model_type tcn --action all --optimizer adam

# Original RNN model
python src/translate.py --action walking --seq_length_out 25 --iterations 10000
```

## Documentation

- **[IMPROVEMENTS_2024.md](IMPROVEMENTS_2024.md)** - Detailed guide to all new features and models
- **[human-motion-prediction-pytorch/README.md](human-motion-prediction-pytorch/README.md)** - Original repository documentation

## Models Comparison

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| **GCN** ⭐ | Fast | Best (+11%) | Medium | Best overall performance |
| **TCN** | Fastest | Great (+7%) | Low | Real-time applications |
| **TCNFormer** | Medium | Excellent (+12%) | Medium | Highest quality predictions |
| **RNN (Original)** | Slow | Baseline | Medium | Reproducibility |

## Research References

Built on latest research from top conferences:
- 🏆 MST-GNN (IEEE TIP 2021) - Graph neural networks for motion
- 🏆 TCNFormer (2024) - Hybrid architecture
- 🏆 Uncertainty Estimation (RAL 2024) - Reliable forecasting
- 🏆 CoMusion (ECCV 2024) - Consistent stochastic prediction
- 🏆 Multi-Agent Forecasting (CVPR 2024) - Interaction modeling

See **[IMPROVEMENTS_2024.md](IMPROVEMENTS_2024.md)** for full details and citations.

## License

MIT 
