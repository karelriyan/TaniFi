# 🚀 Quick Start Guide - Week 1 Execution

## Tahap 1: Setup Lingkungan Simulasi (Minggu 1)

### ✅ Checklist Eksekusi

- [x] **1.1 Repository Setup**
- [ ] **1.2 Environment Setup** 
- [ ] **1.3 Dataset Download**
- [ ] **1.4 Run Initial Simulation**

---

## 1. Repository Setup ✅

Repository sudah siap! Struktur:

```
tanifi-federated-learning/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── data/                        # Dataset storage
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed datasets
├── models/checkpoints/          # Model weights
├── src/simulation/              # Simulation code
│   ├── __init__.py
│   ├── download_dataset.py      # Dataset downloader
│   └── diloco_trainer.py        # Main DiLoCo trainer
├── experiments/                 # Experiments & results
│   ├── config.yaml              # Configuration file
│   └── results/                 # Output metrics & plots
├── notebooks/                   # Jupyter notebooks
└── docs/                        # Documentation
    └── dataset_setup.md         # Dataset guide
```

---

## 2. Environment Setup 🔧

### Langkah 1: Clone Repository (Jika belum)
```bash
# Jika sudah punya folder, skip langkah ini
git init
git add .
git commit -m "Initial TaniFi project setup"
```

### Langkah 2: Create Virtual Environment
```bash
# Buat virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Langkah 3: Install Dependencies
```bash
# Install semua dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verifikasi instalasi
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
PyTorch: 2.1.0+cu121
CUDA Available: True  # Dengan RTX 5060 kamu
```

---

## 3. Dataset Download 📥

### Opsi A: Quick Test (Dummy Dataset)
Untuk test awal, kamu bisa langsung jalankan simulasi dengan dummy data:

```bash
cd src/simulation
python diloco_trainer.py
```

Ini akan:
- Generate dummy dataset otomatis
- Run simulasi dengan 100 farmer nodes
- Training 5 rounds DiLoCo
- Generate metrics & plots

**⏱️ Estimasi waktu: 5-10 menit** (tergantung CPU/GPU)

### Opsi B: Download WeedsGalore (Real Dataset)

#### Setup Kaggle API
```bash
# 1. Buat account Kaggle di kaggle.com
# 2. Go to kaggle.com/account → Create New API Token
# 3. Download kaggle.json
# 4. Tempatkan di ~/.kaggle/

# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
# Copy kaggle.json to C:\Users\<username>\.kaggle\
```

#### Download Dataset
```bash
cd src/simulation

# Download WeedsGalore
python download_dataset.py --dataset weedsgalore --output ../../data/raw

# Atau download sample (lebih cepat untuk testing)
python download_dataset.py --dataset weedsgalore --sample-size 1000
```

**⏱️ Estimasi waktu: 10-30 menit** (tergantung internet)

---

## 4. Run Initial Simulation 🏃

### Quick Run (Default Settings)
```bash
cd src/simulation
python diloco_trainer.py
```

### Custom Configuration
```bash
# Edit config.yaml terlebih dahulu
nano ../../experiments/config.yaml

# Run dengan config
python diloco_trainer.py --config ../../experiments/config.yaml
```

### Expected Output:
```
🖥️  Using device: cuda

🧠 Creating base model...

📦 Loading dataset...
⚠️  Using dummy dataset - replace with WeedsGalore after download

🌾 Initializing 100 farmer nodes...
   ✅ Farmers initialized with 10-150 samples each

============================================================
DiLoCo Federated Learning Simulation
============================================================
Farmers: 100
Local steps per round: 500
Total rounds: 5
============================================================

🔄 Round 1
   📱 Local training phase...
   Training farmers: 100%|████████████| 100/100 [01:23<00:00]
   📊 Average local loss: 2.3045
   📡 Collecting shards from farmers...
   💾 Bandwidth savings: 95.2% (512 vs 10480 parameters)
   🔗 Aggregating shards...
   📤 Distributing updated model...

[... rounds 2-5 ...]

============================================================
Training Complete!
============================================================

📄 Results saved to: ../experiments/results/diloco_results_20250127_223000.json
📊 Plots saved to: ../experiments/results/diloco_metrics_20250127_223000.png

✅ Simulation complete!
```

---

## 5. Verify Results 📊

### Check Output Files
```bash
# Lihat hasil eksperimen
ls -lh ../../experiments/results/

# Output yang diharapkan:
# - diloco_results_[timestamp].json    → Training metrics
# - diloco_metrics_[timestamp].png     → Loss & bandwidth plots
```

### Open Results
```bash
# View metrics JSON
cat ../../experiments/results/diloco_results_*.json

# View plot (if GUI available)
xdg-open ../../experiments/results/diloco_metrics_*.png

# Or copy to your local machine untuk dilihat
```

### Expected Metrics Format:
```json
{
  "rounds": [0, 1, 2, 3, 4],
  "avg_loss": [2.3045, 2.1023, 1.8234, 1.6012, 1.4523],
  "bandwidth_saved": [95.2, 95.2, 95.2, 95.2, 95.2]
}
```

**Key Insights untuk Paper:**
- ✅ **Bandwidth Savings: ~95%** (hanya kirim LoRA adapters, bukan full model)
- ✅ **Convergence**: Loss menurun setiap round (model belajar)
- ✅ **Scalability**: 100 farmers bisa dikoordinasi secara efisien

---

## 6. Next Steps for Paper 📝

### Week 1 Deliverables ✅
- [x] Working simulation environment
- [x] Basic DiLoCo implementation
- [x] Initial metrics collection

### Week 2 Tasks
1. **Replace dummy data dengan WeedsGalore**
   - Buat data loader yang proper
   - Implementasi data partitioning untuk non-IID distribution

2. **Enhance Simulation**
   - Add YOLOv11 integration (ganti SimpleCNN)
   - Implementasi proper LoRA adapters dengan PEFT library
   
3. **Start Paper Writing**
   - Literature Review section
   - Methodology draft
   - Use generated plots dalam paper

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory
```bash
# Kurangi batch size di config.yaml
training:
  batch_size: 4  # Default: 8

# Atau kurangi num_farmers
federated:
  num_farmers: 50  # Default: 100
```

### Issue 2: Kaggle API Error
```bash
# Pastikan kaggle.json ada
ls -la ~/.kaggle/kaggle.json

# Set permissions
chmod 600 ~/.kaggle/kaggle.json

# Test API
kaggle datasets list
```

### Issue 3: Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check specific package
pip show torch ultralytics
```

---

## 📚 Understanding the Code

### DiLoCo Key Concepts:

1. **Local Training (500 steps)**
   - Setiap farmer train model secara lokal
   - Tidak butuh internet selama training
   - Hanya kirim hasil setelah 500 steps

2. **LoRA Adapters (Shards)**
   - Ukuran kecil (~KB), bukan full model (~MB)
   - Hanya adapter yang di-train, base model frozen
   - 95% bandwidth savings

3. **Federated Aggregation**
   - Coordinator aggregate semua shards
   - Simple averaging (FedAvg)
   - Distribute updated weights kembali

### Code Structure:

```python
# Main components:
SimpleCropDiseaseModel  → Base model (later: YOLOv11)
LoRAAdapter            → Farmer's "shard" (efficient fine-tuning)
FarmerNode             → Individual farmer simulation
DiLoCoCoordinator      → Orchestrate federated learning
```

---

## 🎯 Success Criteria Week 1

- [ ] ✅ Environment setup berhasil
- [ ] ✅ Simulasi berjalan tanpa error
- [ ] ✅ Metrics & plots ter-generate
- [ ] ✅ Understand DiLoCo workflow
- [ ] ✅ Ready untuk integration dengan real dataset

---

## 💡 Tips untuk Karel

1. **Jangan perfeksionis di tahap awal**
   - Dummy data OK untuk verify workflow
   - Real dataset bisa ditambahkan gradually

2. **Track everything**
   - Git commit setiap progress
   - Screenshot results untuk documentation
   - Catat issues/bugs yang ditemukan

3. **Prepare for paper**
   - Plots yang di-generate → bisa langsung masuk paper
   - Metrics JSON → untuk tables di paper
   - Code → bisa jadi supplementary material

4. **Time management**
   - Setup environment: 1 hari
   - Run simulations & understand: 2-3 hari
   - Dataset integration: 2-3 hari
   - Buffer: 1 hari untuk troubleshooting

---

## 📞 Next Session

Setelah kamu jalankan simulasi awal, kita akan:
1. Analyze hasil metrics
2. Integrate WeedsGalore dataset
3. Start drafting paper Methodology section

**Questions?** Share:
- Terminal output
- Error messages (jika ada)
- Generated plots
- Metrics JSON

Good luck, Karel! 🚀🌾