# MMTAfrica ([Try out our model](https://huggingface.co/spaces/edaiofficial/mmtafrica))
[Paper](https://aclanthology.org/2021.wmt-1.48/) - [Installation](#installation) - [Example](#example) - [Model checkpoint](#model-checkpoint)  - [Citation](#citation)


This repository contains the official implementation of the MMTAfrica paper ([Emezue & Dossou, WMT 2021](https://aclanthology.org/2021.wmt-1.48/)).

We focus on the task of multilingual machine translation for African languages in the 2021 WMT Shared Task: Large-Scale Multilingual Machine Translation. We introduce MMTAfrica, the first many-to-many multilingual translation system for six African languages: Fon (fon), Igbo (ibo), Kinyarwanda (kin), Swahili/Kiswahili (swa), Xhosa (xho), and Yoruba (yor) and two non-African languages: English (eng) and French (fra). For multilingual translation concerning African languages, we introduce a novel backtranslation and reconstruction objective, BT\&REC, inspired by the random online back translation and T5 modeling framework respectively, to effectively leverage monolingual data. Additionally, we report improvements from MMTAfrica over the FLORES 101 benchmarks (spBLEU gains ranging from +0.58 in Swahili to French to +19.46 in French to Xhosa).

## Installation
To avoid any conflict with your existing Python setup, we suggest to work in a virtual environment:
```
python -m venv mmtenv
source mmtenv/bin/activate
```

Follow these instructions to install MMTAfrica.
```
git clone https://github.com/edaiofficial/mmtafrica.git
cd mmtafrica
pip install -r requirements.txt
```

## Example
```bash
python mmtafrica.py 
```
Consult the arguments [here](https://github.com/edaiofficial/mmtafrica/blob/main/mmtafrica.py#L772-L860).

### Reproducing our paper
Our data for the paper experiments is stored in the `/experiments` folder. To train MMTAfrica from scratch and reproduce our experiemnts, using the data we have in `/experiments`, run
```bash
cd experiments
python ../mmtafrica.py --model_name='mmtafrica' --homepath="<YOUR HOMEPATH>"
```
By default, homepath is the current working directory when you run the code. 

# Model checkpoint
Our model checkpoints is saved [on Google Drive](https://drive.google.com/file/d/1gUINHLRQC06HGGeP211-x3IIr3WS84Iy/view?usp=sharing) and in the [HF Hub](https://huggingface.co/chrisjay/mmtafrica/blob/main/mmt_translation.pt).


## Citation
```
@inproceedings{emezue-dossou-2021-mmtafrica,
    title = "{MMTA}frica: Multilingual Machine Translation for {A}frican Languages",
    author = "Emezue, Chris Chinenye  and
      Dossou, Bonaventure F. P.",
    booktitle = "Proceedings of the Sixth Conference on Machine Translation",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wmt-1.48",
    pages = "398--411",
    abstract = "In this paper, we focus on the task of multilingual machine translation for African languages and describe our contribution in the 2021 WMT Shared Task: Large-Scale Multilingual Machine Translation. We introduce MMTAfrica, the first many-to-many multilingual translation system for six African languages: Fon (fon), Igbo (ibo), Kinyarwanda (kin), Swahili/Kiswahili (swa), Xhosa (xho), and Yoruba (yor) and two non-African languages: English (eng) and French (fra). For multilingual translation concerning African languages, we introduce a novel backtranslation and reconstruction objective, BT{\&}REC, inspired by the random online back translation and T5 modelling framework respectively, to effectively leverage monolingual data. Additionally, we report improvements from MMTAfrica over the FLORES 101 benchmarks (spBLEU gains ranging from +0.58 in Swahili to French to +19.46 in French to Xhosa).",
}
```
