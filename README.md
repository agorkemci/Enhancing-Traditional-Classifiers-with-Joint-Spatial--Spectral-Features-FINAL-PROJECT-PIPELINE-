# Hyperspectral Classification App
## LDA + EMP + CS-KNN Pipeline

### Kurulum

```bash
pip install -r requirements.txt
```

### Çalıştırma

```bash
python main.py
```

### Pipeline Adımları

| # | Adım | Modül |
|---|------|-------|
| 1 | StandardScaler ile normalize | `pipeline/preprocessing.py` |
| 2 | LDA (n_components ayarlanabilir) | `pipeline/preprocessing.py` |
| 3 | LDA → görüntü uzayına geri dönüşüm | `pipeline/preprocessing.py` |
| 4 | EMP (grey opening + closing × 4 ölçek) | `pipeline/preprocessing.py` |
| 5 | Stratified train/test split | `pipeline/pipeline.py` |
| 6 | CS-KNN fit (sınıf ağırlıklı oylama) | `pipeline/csknn.py` |
| 7 | Tam görüntü tahmini + harita üretimi | `pipeline/pipeline.py` |

### Desteklenen Veri Setleri

| Dataset | Data key | GT key |
|---------|----------|--------|
| Indian Pines | `indian_pines` / `indian_pines_corrected` | `indian_pines_gt` |
| Pavia University | `paviaU` | `paviaU_gt` |
| Salinas | `salinas` / `salinas_corrected` | `salinas_gt` |
| Diğer | otomatik algılanır | otomatik algılanır |

### Proje Yapısı

```
hyperspectral_app/
├── main.py                  # Uygulama giriş noktası
├── requirements.txt
├── ui/
│   ├── __init__.py
│   ├── main_window.py       # Ana pencere + worker thread
│   ├── controls_panel.py    # Dataset yükleme & parametre paneli
│   ├── map_viewer.py        # İnteraktif harita görüntüleyici
│   └── metrics_panel.py     # OA/AA/Kappa + confusion matrix
└── pipeline/
    ├── __init__.py
    ├── loader.py            # .mat dosyası yükleme (tek veya çift)
    ├── preprocessing.py     # StandardScaler + LDA + EMP
    ├── csknn.py             # Cost-Sensitive KNN sınıflandırıcı
    ├── metrics.py           # OA, AA, Kappa, per-class accuracy
    └── pipeline.py          # Orchestrator
```
