# 🌱 easy-VLM  
> A minimal yet powerful **vision-language** reproduction for learning & hacking.

---

## 🎯 Objective  
Re-implement a **lightweight** vision-language model to build an intuitive, deep understanding of the mainstream LLM techniques in use today.

---

## 🧬 Origin  
Fork & revival of the original [**minimind-V**](https://github.com/link-to-minimind-V) with:  
- ✏️ **Chinese code comments**  
- 💡 **Personal insights & notes** sprinkled throughout  

---

## 📦 What’s Inside  

| Module | Highlights |
|--------|-----------|
| **Core Building Blocks** | <ul><li>Self-/Cross-Attention</li><li>RMSNorm</li><li>MoE (Mixture of Experts)</li><li>KV-Cache</li><li>RoPE</li></ul> |
| **Training Pipeline** | <ul><li>Pre-training</li><li>Fine-tuning</li></ul> |
| **Multimodal Alignment** | Plug-and-play pre-trained **vision encoders** to tokenize images and inject visual embeddings into the language backbone. |

---

## ⚙️ Hardware Requirements  

| GPU | VRAM | Notes |
|-----|------|-------|
| RTX 4060 Ti | 8 GB | ✅ Trainable with **tuned hyper-params**<br>⚠️ MoE variant shows **noticeable slowdown** |

---

## 🚀 Quick Start  
```bash
git clone https://github.com/your-id/easy-VLM.git
cd easy-VLM
pip install -r requirements.txt
python train.py --config configs/vlm_tiny.yml
