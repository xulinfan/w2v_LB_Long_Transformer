## LittleBeats speaker diarization and vocalization classification recipe
This recipe is based on the SpeechBrain `wav2vec_LittleBeats` recipe.
It adds a new `train_tf_long.py` training entrypoint and a template hyperparameter file while preserving the original recipe structure.

### Overview
- Built on the original SpeechBrain recipe structure.
- Uses the same environment creation and data format as the original SpeechBrain recipe.
- Includes new files under `hparams/` and `scripts/` for this additional model.

### Checkpoint weights
Checkpoint weights are not included in this repository.
The checkpoint folder can be accessed here:

- Checkpoint download link: `https://drive.google.com/drive/folders/1jlOOMWC_XFh5P3-mEIgTVDLgyAtKsdee?usp=sharing`

## Uses

### Prepare data in json format ###
This recipe consumes one classifier label per item. The label is derived from `sp`, `chn`, `fan`, and `man` and the model outputs a single classification score per sample.

To make data compatible with this script, prepare your data similar to the following json format:
```
{
  "sample_data1": {
    "wav_voc": "path/to/your/wav/file1",
    "sp": "SIL",
    "chn": "N",
    "fan": "N",
    "man": "N"
  },
  "sample_data2": {
    "wav_voc": "path/to/your/wav/file2",
    "sp": "CHN",
    "chn": "BAB",
    "fan": "N",
    "man": "N"
  }
}
```

The pipeline maps these fields to a single integer label:
- `SIL`, `CXN` → 0
- `FAN_FAN`, `FAN_CDS`, `FAN_SNG` → 1
- `MAN_MAN`, `MAN_CDS`, `MAN_SNG` → 2
- `CHN_BAB` → 3
- `CHN_CRY` → 4
- `CHN_FUS` → 4

If a sample does not match one of these mappings, it is assigned label `-1` and ignored during training.

Sample json file we used in our experiments can be found in **sample_json/sample_json.json**

### Fine-tune wav2vec2 model on speaker diarization and parent/infant vocalization classification tasks ###
Before running Python script, first run
```
cd recipes/wav2vec_LittleBeats
```

Run the following command to fine-tune wav2vec2 using our developed recipe

```
python scripts/train_tf_long.py hparams/hparams_LL_4300_TF_long.yaml
```

### Paper/BibTex Citation
If you found this recipe or our paper helpful, please cite us as

```
@inproceedings{yourplaceholder,
  author={},
  title={{}},
  booktitle={},
  year={},
}
```