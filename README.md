# PyTorch ile Görüntü Sınıflandırma

Fashion-MNIST veri kümesi üzerinde küçük bir evrişimli sinir ağı (**CNN**) eğiten, okunabilir ve yeniden üretilebilir PyTorch projesi.

## Özellikler

- Eğitim, doğrulama ve test veri akışı
- CPU, CUDA ve Apple Silicon (MPS) desteği
- Rastgelelik tohumlarının sabitlenmesi
- Eğitim kaybı ve doğruluk ölçümü
- En iyi model ağırlıklarını kaydetme
- Tek komutla hızlı deneme modu

## Kurulum

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Eğitim

```bash
python train.py --epochs 5 --batch-size 64
```

Hızlı bir çalışma kontrolü için:

```bash
python train.py --quick --epochs 1
```

Model `artifacts/fashion_cnn.pt` yoluna kaydedilir.

## Model

```text
1×28×28 görüntü
   ↓
Conv(32) → ReLU → MaxPool
   ↓
Conv(64) → ReLU → MaxPool
   ↓
Dropout → Dense(128) → 10 sınıf
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Geliştirme fikirleri

- Veri artırma eklemek
- Karışıklık matrisi üretmek
- TensorBoard ile deney takibi yapmak
- CNN sonucunu bir Vision Transformer ile karşılaştırmak

## Lisans

MIT
