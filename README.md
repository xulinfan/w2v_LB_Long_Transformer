## Open-source SpeechBrain recipe for infant/parent vocalization and children's speech analysis

This repository releases a wav2vec 2.0 long-transformer recipe for speaker diarization and infant vocalization classification, built on top of [SpeechBrain](https://github.com/speechbrain/speechbrain) and trained on LittleBeats home recordings. Our recipe extends [wav2vec_LittleBeats_LENA](https://github.com/jialuli3/wav2vec_LittleBeats_LENA).

### Environment setup

```
git clone https://github.com/xulinfan/w2v_LB_Long_Transformer.git
cd w2v_LB_Long_Transformer
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```

### Recipe

Full usage instructions, data format, and training commands are in the recipe README:

[speechbrain/recipes/wav2vec_LittleBeats/Readme.md](speechbrain/recipes/wav2vec_LittleBeats/Readme.md)

### Citation

If you find this work useful, please cite our paper (coming soon):

```
@inproceedings{yourplaceholder,
  author={},
  title={{}},
  booktitle={},
  year={},
}
```
