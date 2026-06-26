#!/usr/bin/env python3

import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
from collections import defaultdict
import einops

class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        outputs = einops.rearrange(self.modules.wav2vec2(wavs), 'L B T F -> B (L T) F')

        cls_token = einops.rearrange(
            self.modules.cls_token(torch.LongTensor([[0] * outputs.shape[0]]).to(outputs.device)),
            'T B F -> B T F',
        )
        outputs = torch.cat((cls_token, outputs), dim=1)
        outputs = outputs + self.modules.pos_enc(outputs)
        outputs, _ = self.modules.time_mhsa(outputs)
        outputs = self.hparams.layer_norm(outputs)[:, 0]
        outputs = self.modules.output_mlp(self.modules.dnn(outputs))
        return outputs

    def compute_objectives(self, outputs, batch, stage):
        label = batch.label
        batchid = batch.id
        wavs, lens = batch.sig

        predictions = self.hparams.log_softmax(outputs)
        valid_idx = (label != -1).nonzero(as_tuple=True)
        predictions = predictions[valid_idx]
        batchid = [batchid[i] for i in (label != -1).nonzero(as_tuple=False).flatten()]
        wavs = wavs[valid_idx]
        label = label[valid_idx]

        loss = self.hparams.compute_cost(predictions, label)

        if stage != sb.Stage.TRAIN:
            self.error_metrics_kic.append(batch.id, predictions, label)
        return loss

    def fit_batch(self, batch):
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.optimizer.step()

        self.wav2vec2_optimizer.zero_grad()
        self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        predictions = self.compute_forward(batch, stage)
        loss = self.compute_objectives(predictions, batch, stage)
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch=None):
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )
        if stage != sb.Stage.TRAIN:
            self.error_metrics_kic = self.hparams.error_stats_kic()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stats = {
                'loss': stage_loss,
                'error_rate_kappa': 1 - self.error_metrics_kic.summarize('kappa'),
            }

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stats['error_rate_kappa'])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            old_lr_wav2vec2, new_lr_wav2vec2 = self.hparams.lr_annealing_wav2vec2(
                stats['error_rate_kappa']
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )
            self.hparams.train_logger.log_stats(
                {'Epoch': epoch, 'lr': old_lr, 'wave2vec_lr': old_lr_wav2vec2},
                train_stats={'loss': self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stats,
                min_keys=['error_rate_kappa'],
            )
            with open(self.hparams.train_log, 'a') as w:
                self.error_metrics_kic.write_stats(w)

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {'Epoch loaded': self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            with open(self.hparams.train_log, 'a') as w:
                self.error_metrics_kic.write_stats(w)
            with open(self.hparams.output_log, 'w') as w:
                self.error_metrics_kic.write_stats(w)

    def init_optimizers(self):
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable('wav2vec2_opt', self.wav2vec2_optimizer)
            self.checkpointer.add_recoverable('optimizer', self.optimizer)


def dataio_prep(hparams):
    @sb.utils.data_pipeline.takes('wav_voc')
    @sb.utils.data_pipeline.provides('sig')
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    @sb.utils.data_pipeline.takes('sp', 'chn', 'fan', 'man')
    @sb.utils.data_pipeline.provides('label')
    def label_pipeline_flat(sp, chn, fan, man):
        dict_map = {
            'SIL': 0,
            'CXN': 0,
            'MAN_MAN': 2,
            'MAN_CDS': 2,
            'FAN_FAN': 1,
            'FAN_CDS': 1,
            'MAN_SNG': 2,
            'FAN_SNG': 1,
            'CHN_BAB': 3,
            'CHN_CRY': 4,
            'CHN_FUS': 4,
        }
        if sp == 'SIL' or sp == 'CXN':
            label = sp
        elif sp == 'CHN':
            label = f'{sp}_{chn}'
        elif sp == 'FAN':
            label = f'{sp}_{fan}'
        elif sp == 'MAN':
            label = f'{sp}_{man}'

        return dict_map.get(label, -1)

    datasets = {}
    for dataset in ['train', 'valid', 'test']:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f'{dataset}_annotation'],
            replacements={'data_root': hparams['data_folder']},
            dynamic_items=[audio_pipeline, label_pipeline_flat],
            output_keys=['id', 'sig', 'label'],
        )

    return datasets


if __name__ == '__main__':
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams['output_folder'],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = dataio_prep(hparams)

    hparams['wav2vec2'] = hparams['wav2vec2'].to(run_opts['device'])
    hparams['layer_norm'] = hparams['layer_norm'].to(run_opts['device'])

    emo_id_brain = EmoIdBrain(
        modules=hparams['modules'],
        opt_class=hparams['opt_class'],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams['checkpointer'],
    )

    train_loader_options = hparams['dataloader_options'].copy()
    val_loader_options = hparams['dataloader_options'].copy()

    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets['train'],
        valid_set=datasets['valid'],
        train_loader_kwargs=train_loader_options,
        valid_loader_kwargs=val_loader_options,
    )

    test_stats = emo_id_brain.evaluate(
        test_set=datasets['test'],
        min_key='error_rate_kappa',
        test_loader_kwargs=hparams['dataloader_options'],
    )
