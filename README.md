# Attention Is All You Need - Implementation

A from-scratch implementation of the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## 🎯 Project Goals

This project implements the complete Transformer architecture with:
- **Educational Focus**: Understanding every component by building from scratch
- **Production-Ready Code**: Clean, modular, and well-tested implementation
- **Reproducible Results**: Proper seed management and configuration system
- **Modern PyTorch**: Leveraging latest PyTorch features including Apple Silicon MPS support

## 🏗️ Project Structure

```
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Transformer model components
│   ├── train/          # Training loops and utilities
│   ├── inference/      # Inference and generation code
│   └── utils/          # Configuration, logging, and utilities
├── tests/              # Unit tests and smoke tests
├── config/             # Configuration files
└── requirements.txt    # Dependencies
```

## 🚀 Quick Start

### Setup

```bash
# Clone and navigate to the project
cd att-is-all-we-need

# Install dependencies
pip install -r requirements.txt

# Verify setup
python -m tests.smoke
```

### Configuration

The project uses YAML configuration files. See `config/default.yaml` for all available options:

- **Model Architecture**: d_model, n_heads, n_layers, etc.
- **Training Settings**: batch_size, learning_rate, optimizer settings
- **System Settings**: device selection, random seed, logging

### Running Tests

```bash
# Run smoke test to verify setup
python -m tests.smoke

# Run all tests
pytest tests/
```

## 🔧 System Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Hardware**: CPU, CUDA GPU, or Apple Silicon (MPS)

### Apple Silicon Support

This implementation includes full support for Apple Silicon Macs using Metal Performance Shaders (MPS):

```yaml
system:
  device: "auto"  # Automatically detects and uses MPS on Apple Silicon
```

## 📋 Implementation Roadmap

- [x] **Step 0**: Project skeleton, configuration, logging, seed management
- [x] **Step 1**: LayerNorm and stable softmax
- [x] **Step 2**: Scaled dot-product attention
- [x] **Step 3**: Multi-Head Attention
- [x] **Step 4**: Positional encodings (sinusoidal & learned)
- [x] **Step 5**: Position-wise Feed-Forward Network
- [ ] **Step 6**: Encoder layer & stack
- [ ] **Step 7**: Decoder layer & stack
- [ ] **Step 8**: Embeddings and weight tying
- [ ] **Step 9**: Data pipeline & tokenization
- [ ] **Step 10**: Training objective and label smoothing
- [ ] **Step 11**: Optimizer and learning rate schedule
- [ ] **Step 12**: Training loop and mixed precision
- [ ] **Step 13**: Inference with beam search
- [ ] **Step 14**: Evaluation metrics
- [ ] **Step 15**: Reproducibility and ablations

## 🏃‍♂️ Getting Started

### Smoke Test Output

After setup, you should see output like:

```
🚀 Starting Attention Is All You Need - Smoke Test
============================================================
✓ Configuration loaded successfully
✓ Logger setup successfully
✓ Random seed set to: 42
✓ Device selected: mps

📊 System Information:
------------------------------
PyTorch version: 2.8.0
NumPy version: 2.3.2
CUDA available: False
MPS available: True

✅ Project setup is working correctly
```

## 📖 Key Features

### Configuration System
- YAML-based configuration with hierarchical structure
- Command-line argument overrides
- Model size presets (small, base, large)

### Logging & Monitoring
- Structured logging with configurable levels
- TensorBoard integration for training visualization
- Progress tracking and metrics logging

### Reproducibility
- Fixed random seeds across Python, NumPy, and PyTorch
- Deterministic algorithms for consistent results
- Configuration versioning and experiment tracking

### Device Support
- Automatic device detection (CPU/CUDA/MPS)
- Apple Silicon MPS optimization
- Memory usage monitoring

## 🔬 Testing

The project includes comprehensive testing:

- **Smoke Tests**: Verify basic setup and functionality
- **Unit Tests**: Test individual components
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Memory and speed benchmarks

## 📚 Learning Resources

This implementation follows the original paper closely while incorporating modern best practices:

- **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Code Comments**: Detailed explanations of each component
- **Test Cases**: Examples showing expected shapes and behaviors
- **Configuration**: Well-documented parameters and their effects
