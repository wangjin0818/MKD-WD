"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for VCR
"""
import argparse
import json
import os
from os.path import abspath, dirname, exists, join
from time import time

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam, Adamax

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data import (TokenBucketSampler, PrefetchLoader, DetectFeatLmdb,
                  VcrTxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens,
                  VcrDataset, VcrEvalDataset,
                  vcr_collate, vcr_eval_collate, )
from model.vcr import UniterForVisualCommonsenseReasoning
from optim import AdamW, get_lr_sched

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import BUCKET_SIZE, IMG_DIM
from KD_loss import distillation_loss, patience_loss
from distiller_zoo import VIDLoss
from MSD_loss import msd_loss, get_p_sum

NUM_SPECIAL_TOKENS = 81
student_train = True
import os
from pyemd import emd_with_flow
from torch.nn.modules.loss import MSELoss
import numpy as np
# from torchsummary import summary

def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    if is_train:
        sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                     batch_size=batch_size, droplast=is_train)
        dataloader = DataLoader(dataset, batch_sampler=sampler,
                                num_workers=opts.n_workers,
                                pin_memory=opts.pin_mem, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=opts.n_workers, shuffle=False,
                                pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def build_optimizer(model, opts):
    """ vqa linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'vcr_output' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'vcr_output' in n]
    # print(param_optimizer)
    # print(param_top)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    # print(optimizer_grouped_parameters)
    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer

def get_new_layer_weight(trans_matrix, distance_matrix, stu_layer_num, tea_layer_num, T, att_student_weight, att_teacher_weight, rep_student_weight, rep_teacher_weight, type_update='att'):
    if type_update == 'att':
        student_layer_weight = np.copy(att_student_weight)
        teacher_layer_weight = np.copy(att_teacher_weight)
    else:
        student_layer_weight = np.copy(rep_student_weight)
        teacher_layer_weight = np.copy(rep_teacher_weight)

    distance_matrix = distance_matrix.detach().cpu().numpy().astype('float64')
    trans_weight = np.sum(trans_matrix * distance_matrix, -1)
    # logger.info('student_trans_weight:{}'.format(trans_weight))
    # new_student_weight = torch.zeros(stu_layer_num).cuda()
    for i in range(stu_layer_num):
        student_layer_weight[i] = trans_weight[i] / student_layer_weight[i]
    weight_sum = np.sum(student_layer_weight)
    for i in range(stu_layer_num):
        if student_layer_weight[i] != 0:
            student_layer_weight[i] = weight_sum / student_layer_weight[i]

    trans_weight = np.sum(np.transpose(trans_matrix) * distance_matrix, -1)
    for j in range(tea_layer_num):
        teacher_layer_weight[j] = trans_weight[j + stu_layer_num] / teacher_layer_weight[j]
    weight_sum = np.sum(teacher_layer_weight)
    for i in range(tea_layer_num):
        if teacher_layer_weight[i] != 0:
            teacher_layer_weight[i] = weight_sum / teacher_layer_weight[i]

    student_layer_weight = student_layer_weight / np.sum(student_layer_weight)
    teacher_layer_weight = teacher_layer_weight / np.sum(teacher_layer_weight)

    if type_update == 'att':
        att_student_weight = student_layer_weight
        att_teacher_weight = teacher_layer_weight
    else:
        rep_student_weight = student_layer_weight
        rep_teacher_weight = teacher_layer_weight

    return  student_layer_weight, teacher_layer_weight

def transformer_loss(student_reps, teacher_reps, device, loss_mse, global_step, opt, att_student_weight, rep_student_weight, att_teacher_weight, rep_teacher_weight, T=1, update_weight = True, use_rep = True, embedding_emd = False, use_att = False, student_atts = None, teacher_atts = None):
    def embedding_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j]
                tmp_loss = loss_mse(student_rep, teacher_rep)
                # tmp_loss = torch.nn.functional.normalize(tmp_loss, p=2, dim=2)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        # trans_matrix = trans_matrix
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix

    def emd_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j]
                tmp_loss = loss_mse(student_rep, teacher_rep)
                # tmp_loss = torch.nn.functional.normalize(tmp_loss, p=2, dim=2)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        # trans_matrix = trans_matrix
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix

    def emd_att_loss(student_atts, teacher_atts, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):

        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_att = student_atts[i]
            for j in range(tea_layer_num):
                teacher_att = teacher_atts[j]
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                          student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                          teacher_att)

                tmp_loss = loss_mse(student_att, teacher_att)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss
        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        att_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return att_loss, trans_matrix, distance_matrix

    stu_layer_num = len(student_reps)
    tea_layer_num = len(teacher_reps)
    if use_att:
        att_loss, att_trans_matrix, att_distance_matrix = \
            emd_att_loss(student_atts, teacher_atts, att_student_weight, att_teacher_weight,
                         stu_layer_num, tea_layer_num, device, loss_mse)
        if update_weight:
            att_student_weight, att_teacher_weight = get_new_layer_weight(att_trans_matrix, att_distance_matrix, stu_layer_num, tea_layer_num, att_student_weight = att_student_weight, att_teacher_weight = att_teacher_weight, rep_student_weight=rep_student_weight, rep_teacher_weight=rep_teacher_weight,T=T)
        att_loss = att_loss.to(device)
    else:
        att_loss = torch.tensor(0)
    if use_rep:
        if embedding_emd:
            rep_loss, rep_trans_matrix, rep_distance_matrix = \
                embedding_rep_loss(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                             stu_layer_num+1, tea_layer_num+1, device, loss_mse)
            if update_weight:
                rep_student_weight, rep_teacher_weight = get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num+1, tea_layer_num+1, att_student_weight = att_student_weight, att_teacher_weight = att_teacher_weight, rep_student_weight=rep_student_weight, rep_teacher_weight=rep_teacher_weight,T=T, type_update='xx')
        else:
            rep_loss, rep_trans_matrix, rep_distance_matrix = \
                emd_rep_loss(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                             stu_layer_num, tea_layer_num, device, loss_mse)

            if update_weight:
                rep_student_weight, rep_teacher_weight = get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num, att_student_weight = att_student_weight, att_teacher_weight = att_teacher_weight, rep_student_weight=rep_student_weight, rep_teacher_weight=rep_teacher_weight,T=T, type_update='xx')
        rep_loss = rep_loss.to(device)
    else:
        rep_loss = torch.tensor(0)


    if not opt.seperate:
        student_weight = np.mean(np.stack([att_student_weight, rep_student_weight]), 0)
        teacher_weight = np.mean(np.stack([att_teacher_weight, rep_teacher_weight]), 0)
        if global_step % opt.valid_steps == 0:
            LOGGER.info('all_student_weight:{}'.format(student_weight))
            LOGGER.info('all_teacher_weight:{}'.format(teacher_weight))
        att_student_weight = student_weight
        att_teacher_weight = teacher_weight
        rep_student_weight = student_weight
        rep_teacher_weight = teacher_weight
    else:
        if global_step % opt.valid_steps == 0:
            LOGGER.info('att_student_weight:{}'.format(att_student_weight))
            LOGGER.info('att_teacher_weight:{}'.format(att_teacher_weight))
            LOGGER.info('rep_student_weight:{}'.format(rep_student_weight))
            LOGGER.info('rep_teacher_weight:{}'.format(rep_teacher_weight))

#       att_student_weight = att_student_weight / np.sum(att_student_weight)
#       att_teacher_weight = att_teacher_weight / np.sum(att_teacher_weight)

#       rep_student_weight = rep_student_weight / np.sum(rep_student_weight)
#       rep_teacher_weight = rep_teacher_weight / np.sum(rep_student_weight)
    if use_att and not use_rep:
        return att_loss, att_student_weight, att_teacher_weight
    elif use_rep and not use_att:
        return rep_loss, rep_student_weight, rep_teacher_weight, rep_trans_matrix, rep_distance_matrix
    else:
        return att_loss, rep_loss, rep_student_weight, rep_teacher_weight, att_student_weight, att_teacher_weight
def load_img_feat(db_list, all_img_dbs, opts):
    db_ = db_list.split(";")
    assert len(db_) <= 2, "More than two img_dbs found"
    gt_db_path, db_path = "", ""
    for d in db_:
        if "gt" in d:
            gt_db_path = d
        else:
            db_path = d
    if gt_db_path != "":
        img_db_gt = DetectFeatLmdb(
            gt_db_path, -1, opts.max_bb, opts.min_bb, 100,
            opts.compressed_db)
        all_img_dbs.path2imgdb[gt_db_path] = img_db_gt
    else:
        img_db_gt = None
    img_db = all_img_dbs[db_path] if db_path != "" else None
    all_img_dbs.path2imgdb[db_path] = img_db
    return img_db, img_db_gt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
        device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    # load DBs and image dirs
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)
    # train
    LOGGER.info(f"Loading Train Dataset "
                f"{opts.train_txt_dbs}, {opts.train_img_dbs}")
    train_datasets = []
    for txt_path, img_path in zip(opts.train_txt_dbs, opts.train_img_dbs):
        img_db, img_db_gt = load_img_feat(img_path, all_img_dbs, opts)
        qa_txt_db = VcrTxtTokLmdb(txt_path, opts.max_txt_len, task="qa")
        qar_txt_db = VcrTxtTokLmdb(txt_path, opts.max_txt_len, task="qar")
        train_datasets.append(
            VcrDataset(qa_txt_db, img_db_gt=img_db_gt, img_db=img_db))
        train_datasets.append(
            VcrDataset(qar_txt_db, img_db_gt=img_db_gt, img_db=img_db))
    train_dataset = ConcatDatasetWithLens(train_datasets)
    train_dataloader = build_dataloader(train_dataset, vcr_collate, True, opts)
    # val
    LOGGER.info(f"Loading Val Dataset {opts.val_txt_db}, {opts.val_img_db}")
    val_img_db, val_img_db_gt = load_img_feat(opts.val_img_db, all_img_dbs, opts)
    val_txt_db = VcrTxtTokLmdb(opts.val_txt_db, -1)
    val_dataset = VcrEvalDataset(
        "val", val_txt_db, img_db=val_img_db, img_db_gt=val_img_db_gt)
    val_final_dataset = VcrEvalDataset(
        "test", val_txt_db, img_db=val_img_db, img_db_gt=val_img_db_gt)
    val_dataloader = build_dataloader(val_dataset, vcr_eval_collate,
                                      False, opts)
    val_final_dataloader = build_dataloader(
        val_final_dataset, vcr_eval_collate,
        False, opts)

    # Prepare model
    if opts.checkpoint and opts.checkpoint_from == "pretrain":
        ckpt = torch.load(opts.checkpoint)
        checkpoint = {k.replace('bert', 'uniter'): v for k, v in ckpt.items()}
    else:
        checkpoint = {}

    all_dbs = opts.train_txt_dbs + [opts.val_txt_db]
    toker = json.load(open(f'{all_dbs[0]}/meta.json'))['bert']
    assert all(toker == json.load(open(f'{db}/meta.json'))['bert']
               for db in all_dbs)
    if student_train == True:
        model = UniterForVisualCommonsenseReasoning.from_pretrained(
            opts.model_config, state_dict={},
            img_dim=IMG_DIM)
        model.init_type_embedding()
        model.init_word_embedding(NUM_SPECIAL_TOKENS)
        ckpt_file = opts.checkpoint
        ckpt = torch.load(ckpt_file)
        checkpoint = {k.replace('bert', 'uniter'): v for k, v in ckpt.items()}
        state_dict = checkpoint.get('model_state', checkpoint)
        matched_state_dict = {}
        unexpected_keys = set()
        missing_keys = set()
        for name, param in model.named_parameters():
            missing_keys.add(name)
        for key, data in state_dict.items():
            if key in missing_keys:
                matched_state_dict[key] = data
                missing_keys.remove(key)
            else:
                unexpected_keys.add(key)
        LOGGER.info(f"Unexpected_keys: {list(unexpected_keys)}")
        LOGGER.info(f"Missing_keys: {list(missing_keys)}")
        model.load_state_dict(matched_state_dict, strict=False)
        # if opts.fp16:
        #     model = amp.initialize(model, enabled=True, opt_level='O2')
        model.to(device)

        student_checkpoint = {}
        student_model = UniterForVisualCommonsenseReasoning.from_pretrained(
            opts.student_model_config, student_checkpoint, img_dim=IMG_DIM)
        student_model.init_type_embedding()
        student_model.init_word_embedding(NUM_SPECIAL_TOKENS)

        student_checkpoint = torch.load(opts.student_checkpoint)
        student_checkpoint = {k.replace('bert', 'uniter'): v for k, v in student_checkpoint.items()}
        student_state_dict = checkpoint.get('model_state', student_checkpoint)
        student_matched_state_dict = {}
        student_unexpected_keys = set()
        student_missing_keys = set()
        for name, param in student_model.named_parameters():
            student_missing_keys.add(name)
        for key, data in student_state_dict.items():
            if key in student_missing_keys:
                student_matched_state_dict[key] = data
                student_missing_keys.remove(key)
            else:
                student_unexpected_keys.add(key)
        print("Unexpected_keys:", list(student_unexpected_keys))
        print("Missing_keys:", list(student_missing_keys))
        student_model.load_state_dict(student_matched_state_dict, strict=False)
        del student_checkpoint
        student_model.to(device)
    else:
        model = UniterForVisualCommonsenseReasoning.from_pretrained(
            opts.model_config, checkpoint, img_dim=IMG_DIM)
        model.init_type_embedding()
        model.init_word_embedding(NUM_SPECIAL_TOKENS)
        if opts.checkpoint_from == "vcr_pretrain":
            ckpt = torch.load(opts.checkpoint)
            checkpoint = {k.replace('bert', 'uniter'): v for k, v in ckpt.items()}
            state_dict = checkpoint.get('model_state', checkpoint)
            matched_state_dict = {}
            unexpected_keys = set()
            missing_keys = set()
            for name, param in model.named_parameters():
                missing_keys.add(name)
            for key, data in state_dict.items():
                if key in missing_keys:
                    matched_state_dict[key] = data
                    missing_keys.remove(key)
                else:
                    unexpected_keys.add(key)
            print("Unexpected_keys:", list(unexpected_keys))
            print("Missing_keys:", list(missing_keys))
            model.load_state_dict(matched_state_dict, strict=False)
        del checkpoint
        model.to(device)
    # make sure every process has same model parameters in the beginning

    broadcast_tensors([p.data for p in student_model.parameters()], 0)
    set_dropout(student_model, opts.dropout)

    t_Total_params = 0
    t_Embedding_params = 0

    s_Total_params = 0
    s_Embedding_params = 0
    def cal_para_num(model, Total_params, Embedding_params):
        for (param_name, param) in model.named_parameters():
            if 'embeddings' not in param_name and 'img_embeddings' not in param_name:
                mulValue = np.prod(param.size())
                Total_params += mulValue

            else:
                E_mulValue = np.prod(param.size())
                Embedding_params += E_mulValue
        print("Trim_params:", Total_params)
        print("Embedding_params:", Embedding_params)
        print("Total_parameter:",Total_params + Embedding_params)
    # Prepare optimizer
    optimizer = build_optimizer(student_model, opts)

    cal_para_num(model, t_Total_params, t_Embedding_params)
    cal_para_num(student_model, s_Total_params, s_Embedding_params)
    # print(student_model.named_parameters())
    student_model, optimizer = amp.initialize(student_model, optimizer,
                                              enabled=opts.fp16, opt_level='O2')
    global_step = 0
    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        os.makedirs(join(opts.output_dir, 'results'))  # store VQA predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataset) * hvd.size())
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    if student_train == True:
        student_model.train()
    else:
        model.train()
    n_examples = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()

    loss_mse = MSELoss()
    # global att_student_weight, rep_student_weight, att_teacher_weight, rep_teacher_weight

    # 初始化BMD权重矩阵
    T_num_hidden_layers = 12
    S_num_hidden_layers = 6
    img_txt_att_student_weight = np.ones(S_num_hidden_layers) / S_num_hidden_layers
    img_txt_rep_student_weight = np.ones(S_num_hidden_layers) / S_num_hidden_layers
    img_txt_att_teacher_weight = np.ones(T_num_hidden_layers) / T_num_hidden_layers
    img_txt_rep_teacher_weight = np.ones(T_num_hidden_layers) / T_num_hidden_layers

    img_att_student_weight = np.ones(S_num_hidden_layers) / S_num_hidden_layers
    img_rep_student_weight = np.ones(S_num_hidden_layers) / S_num_hidden_layers
    img_att_teacher_weight = np.ones(T_num_hidden_layers) / T_num_hidden_layers
    img_rep_teacher_weight = np.ones(T_num_hidden_layers) / T_num_hidden_layers

    txt_att_student_weight = np.ones(S_num_hidden_layers) / S_num_hidden_layers
    txt_rep_student_weight = np.ones(S_num_hidden_layers) / S_num_hidden_layers
    txt_att_teacher_weight = np.ones(T_num_hidden_layers) / T_num_hidden_layers
    txt_rep_teacher_weight = np.ones(T_num_hidden_layers) / T_num_hidden_layers

    if not exists(f'{opts.output_dir}/tran_numpy'):
        os.mkdir(f'{opts.output_dir}/tran_numpy')
    while True:
        for step, batch in enumerate(train_dataloader):
            n_examples += batch['input_ids'].size(0)
            # print("n_examples:", n_examples)
            if student_train == True:
                # answer_scores, all_encoder_layers = model(batch, compute_loss=True)
                with torch.no_grad():
                    t_img_all_encoder_layers, t_img_answer_scores, t_txt_all_encoder_layers, t_txt_answer_scores, t_img_txt_all_encoder_layers, t_img_txt_answer_scores = model(
                        batch, compute_loss=False)
                s_img_all_encoder_layers, s_img_answer_scores, s_txt_all_encoder_layers, s_txt_answer_scores, s_img_txt_all_encoder_layers, s_img_txt_answer_scores = student_model(
                    batch, compute_loss=False)

                # start1 = time()
                # with torch.no_grad():
                #     t_img_txt_answer_scores, t_img_txt_all_encoder_layers = model(batch, compute_loss=False, is_val = True)
                # end1 = time()
                # print("teacher_inference_time:", end1 - start1)
                # # summary(model, )
                # start2 = time()
                # s_img_txt_answer_scores, s_img_txt_all_encoder_layers= student_model(batch, compute_loss=False, is_val = True)
                # end2 = time()
                # print("student_inference_time:", end2 - start2)
                # exit()
                alpha = 0.1
                beta = 0.6

                T = 1
                # loss_dl = distillation_loss(student_answer_score, batch['targets'], answer_scores, 5, alpha=alpha)

                loss_dl = msd_loss(s_img_answer_scores, t_img_answer_scores, s_txt_answer_scores,
                                   t_txt_answer_scores, s_img_txt_answer_scores, t_img_txt_answer_scores,
                                   batch['targets'], alpha, T, task='vcr')

                img_p_sum, img_rep_student_weight, img_rep_teacher_weight, img_rep_trans_matrix, img_rep_distance_matrix = transformer_loss(
                    student_reps=s_img_all_encoder_layers,
                    teacher_reps=t_img_all_encoder_layers, device=device, loss_mse=loss_mse,
                    global_step=global_step, opt=opts, T=T, update_weight=True, use_rep=True,
                    embedding_emd=False, use_att=False, student_atts=None,
                    teacher_atts=None, att_student_weight=img_att_student_weight,
                    rep_student_weight=img_rep_student_weight, att_teacher_weight=img_att_teacher_weight,
                    rep_teacher_weight=img_rep_teacher_weight)
                txt_p_sum, txt_rep_student_weight, txt_rep_teacher_weight, txt_rep_trans_matrix, txt_rep_distance_matrix = transformer_loss(
                    student_reps=s_txt_all_encoder_layers,
                    teacher_reps=t_txt_all_encoder_layers, device=device, loss_mse=loss_mse,
                    global_step=global_step, opt=opts, T=T, update_weight=True, use_rep=True,
                    embedding_emd=False, use_att=False, student_atts=None,
                    teacher_atts=None, att_student_weight=txt_att_student_weight,
                    rep_student_weight=txt_rep_student_weight, att_teacher_weight=txt_att_teacher_weight,
                    rep_teacher_weight=txt_rep_teacher_weight)
                img_txt_p_sum, img_txt_rep_student_weight, img_txt_rep_teacher_weight, img_txt_rep_trans_matrix, img_txt_rep_distance_matrix = transformer_loss(
                    student_reps=s_img_txt_all_encoder_layers,
                    teacher_reps=t_img_txt_all_encoder_layers, device=device,
                    loss_mse=loss_mse, global_step=global_step, opt=opts, T=T,
                    update_weight=True, use_rep=True, embedding_emd=False, use_att=False,
                    student_atts=None, teacher_atts=None, att_student_weight=img_txt_att_student_weight,
                    rep_student_weight=img_txt_rep_student_weight, att_teacher_weight=img_txt_att_teacher_weight,
                    rep_teacher_weight=img_txt_rep_teacher_weight)


                if global_step % opts.valid_steps == 0:
                    np.save(f'{opts.output_dir}/tran_numpy/{global_step}_img_rep_trans_matrix', np.asarray(img_rep_trans_matrix))
                    np.save(f'{opts.output_dir}/tran_numpy/{global_step}_img_rep_distance_matrix', img_rep_distance_matrix.detach().cpu().numpy().astype('float64'))
                    # print("img_rep_distance_matrix:", img_rep_distance_matrix)
                    # exit()
                    np.save(f'{opts.output_dir}/tran_numpy/{global_step}_txt_rep_trans_matrix',
                            np.asarray(txt_rep_trans_matrix))
                    np.save(f'{opts.output_dir}/tran_numpy/{global_step}_txt_rep_distance_matrix', txt_rep_distance_matrix.detach().cpu().numpy().astype('float64'))

                    np.save(f'{opts.output_dir}/tran_numpy/{global_step}_img_txt_rep_trans_matrix',
                            np.asarray(img_txt_rep_trans_matrix))
                    np.save(f'{opts.output_dir}/tran_numpy/{global_step}_img_txt_rep_distance_matrix', img_txt_rep_distance_matrix.detach().cpu().numpy().astype('float64'))


                # img_p_sum = get_p_sum(t_img_all_encoder_layers, s_img_all_encoder_layers, mi = 'mse', last = True)
                # txt_p_sum = get_p_sum(t_txt_all_encoder_layers, s_txt_all_encoder_layers, mi = 'mse', last = True)
                # img_txt_p_sum = get_p_sum(t_img_txt_all_encoder_layers, s_img_txt_all_encoder_layers, mi = 'mse', last = True)
                p_sum = 0.25 * img_p_sum + 0.25 * txt_p_sum + 0.5 * img_txt_p_sum
                # loss = distillation_loss(s_img_txt_answer_scores, batch['targets'], img_txt_answer_scores, 10,
                #                                               alpha=alpha, task = 'vcr')
                # p_sum = img_txt_p_sum

            loss = (1 - beta) * loss_dl + beta * p_sum.half()

            delay_unscale = (step + 1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in student_model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())

            # ============================ End ==========================

            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for i, param_group in enumerate(optimizer.param_groups):
                    if i == 0 or i == 1:
                        param_group['lr'] = lr_this_step * opts.lr_mul
                    elif i == 2 or i == 3:
                        param_group['lr'] = lr_this_step
                    else:
                        raise ValueError()
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 100 == 0:
                    # monitor training throughput
                    LOGGER.info(f'============Step {global_step}=============')
                    tot_ex = sum(all_gather_list(n_examples))
                    ex_per_sec = int(tot_ex / (time() - start))
                    LOGGER.info(f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)
                    LOGGER.info('===========================================')

                if global_step % opts.valid_steps == 0:
                    val_log, results = validate(
                        student_model, val_dataloader)
                    TB_LOGGER.log_scaler_dict(val_log)
                    model_saver.save(student_model, global_step)
            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"finished {n_epoch} epochs")
    if global_step % opts.valid_steps != 0:
        val_log, results = validate(
            student_model, val_dataloader)
        TB_LOGGER.log_scaler_dict(val_log)
    val_log, results = validate(student_model, val_final_dataloader)
    with open(f'{opts.output_dir}/results/'
              f'results_{global_step}_final_qa_qar_'
              f'rank{rank}.json', 'w') as f:
        json.dump(results, f)
    TB_LOGGER.log_scaler_dict(val_log)
    model_saver.save(student_model, global_step)


def compute_accuracies(out_qa, labels_qa, out_qar, labels_qar):
    outputs_qa = out_qa.max(dim=-1)[1]
    outputs_qar = out_qar.max(dim=-1)[1]
    matched_qa = outputs_qa.squeeze() == labels_qa.squeeze()
    matched_qar = outputs_qar.squeeze() == labels_qar.squeeze()
    matched_joined = matched_qa & matched_qar
    n_correct_qa = matched_qa.sum().item()
    n_correct_qar = matched_qar.sum().item()
    n_correct_joined = matched_joined.sum().item()
    return n_correct_qa, n_correct_qar, n_correct_joined


@torch.no_grad()
def validate(model, val_loader):
    if hvd.rank() == 0:
        val_pbar = tqdm(total=len(val_loader))
    else:
        val_pbar = NoOp()
    LOGGER.info("start running validation...")
    model.eval()
    val_qa_loss, val_qar_loss = 0, 0
    tot_qa_score, tot_qar_score, tot_score = 0, 0, 0
    n_ex = 0
    st = time()
    results = {}
    for i, batch in enumerate(val_loader):
        scores = model(
            batch, compute_loss=False, is_val=True)[0]
        qa_targets = batch['qa_targets']
        qar_targets = batch['qar_targets']
        qids = batch['qids']
        scores = scores[:, 1:]
        scores = scores.view(len(qids), -1)
        vcr_qa_loss = F.cross_entropy(
            scores[:, :4], qa_targets.squeeze(-1), reduction="sum")
        if scores.shape[1] > 8:
            qar_index = [4 + answer_ind.item() * 4 + i for answer_ind in qa_targets
                         for i in range(4)]
            qar_scores = scores[:, qar_index]
        else:
            qar_scores = scores[:, 4:]
        vcr_qar_loss = F.cross_entropy(
            qar_scores, qar_targets.squeeze(-1), reduction="sum")
        val_qa_loss += vcr_qa_loss.item()
        val_qar_loss += vcr_qar_loss.item()
        curr_qa_score, curr_qar_score, curr_score = compute_accuracies(
            scores[:, :4], qa_targets, qar_scores, qar_targets)
        tot_qar_score += curr_qar_score
        tot_qa_score += curr_qa_score
        tot_score += curr_score
        for qid, score in zip(qids, scores):
            results[qid] = score.cpu().tolist()
        n_ex += len(qids)
        val_pbar.update(1)
    val_qa_loss = sum(all_gather_list(val_qa_loss))
    val_qar_loss = sum(all_gather_list(val_qar_loss))
    tot_qa_score = sum(all_gather_list(tot_qa_score))
    tot_qar_score = sum(all_gather_list(tot_qar_score))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time() - st
    val_qa_loss /= n_ex
    val_qar_loss /= n_ex
    val_qa_acc = tot_qa_score / n_ex
    val_qar_acc = tot_qar_score / n_ex
    val_acc = tot_score / n_ex
    val_log = {'valid/vcr_qa_loss': val_qa_loss,
               'valid/vcr_qar_loss': val_qar_loss,
               'valid/acc_qa': val_qa_acc,
               'valid/acc_qar': val_qar_acc,
               'valid/acc': val_acc,
               'valid/ex_per_s': n_ex / tot_time}
    model.train()
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score_qa: {val_qa_acc * 100:.2f} "
                f"score_qar: {val_qar_acc * 100:.2f} "
                f"score: {val_acc * 100:.2f} ")
    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seperate", default=False, type=bool)
    # Required parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model")
    parser.add_argument("--checkpoint_from",
                        default='pretrain', type=str,
                        choices=['pretrain', 'vcr_pretrain'],
                        help="which setting is checkpoint from")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_mul", default=10.0, type=float,
                        help="multiplier for top layer lr")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    # adversarial training related
    parser.add_argument('--adv_training', action='store_true',
                        help="Whether to use adversarial training or not")
    parser.add_argument("--adv_modality", default=['text'],
                        help="add pertubation on text or image modality")
    parser.add_argument('--adv_lr_txt', type=float, default=0)
    parser.add_argument('--adv_lr_img', type=float, default=0)
    parser.add_argument('--adv_steps', type=int, default=1,
                        help="should be at least 1")
    parser.add_argument('--norm_type', type=str, default="l2",
                        choices=["l2", "linf"])
    parser.add_argument('--adv_max_norm', type=float, default=0,
                        help="set to 0 to be unlimited")
    parser.add_argument('--adv_kl_weight', type=float, default=0,
                        help="set to 0 to be unlimited")

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
