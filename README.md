## Open-source SpeechBrain recipe for infant/parent vocalization and children's speech analysis

This repository releases a wav2vec 2.0 long-transformer recipe for speaker diarization and parent/infant vocalization classification, built on top of [SpeechBrain](https://github.com/speechbrain/speechbrain) and trained on LittleBeats home recordings.

### Environment setup

```
git clone <your-repo-url>
cd wav2vec_LittleBeats_LENA-main
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```

### Recipe

Full usage instructions, data format, and training commands are in the recipe README:

[speechbrain/recipes/wav2vec_LittleBeats/Readme.md](speechbrain/recipes/wav2vec_LittleBeats/Readme.md)

### Citation

If you find this work useful, please cite our paper:

```
@inproceedings{yourplaceholder,
  author={},
  title={{}},
  booktitle={},
  year={},
}
```
