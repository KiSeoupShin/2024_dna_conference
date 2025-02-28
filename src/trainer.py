# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import pdb
import wandb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
from third_party.open_clip.simple_tokenizer import SimpleTokenizer
from utils import is_master


def get_loss(model, images, texts, loss_img, loss_txt, args, data_identifier=-1):
    if data_identifier == 1:
        # ImageNet dataset
        image_features, text_features, logit_scale = model(images, texts, extra=True)
    else:
        image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # Image loss.
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss


def get_text_features(model, token_features, args):
    text = tokenize("a photo of")
    text = text.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    text_features = model.encode_text_img(text, token_features)
    return text_features

def get_text_features_add_caption(model, token_features, texts, args):
    text = tokenize("a photo of")
    caption = tokenize([t for t in texts])
    text = text.cuda(args.gpu, non_blocking=True)
    caption = caption.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    text_features = model.encode_text_img_cap(text, token_features, caption)
    return text_features

def get_text_features_alpha(model, texts, token_features, args):
    texts = tokenize(texts)
    texts = texts.cuda(args.gpu, non_blocking=True)
    text_features = model.encode_text_img(texts, token_features)
    return text_features

def get_text_features_only_caption(model, texts, args):
    texts = ["a photo of " + t for t in texts]
    texts = tokenize(texts)
    texts = texts.cuda(args.gpu, non_blocking=True)
    text_features = model.encode_text(texts)
    return text_features

def get_text_class_features(model, text_features, class_features, args):
    text = tokenize("a photo of")
    text = text.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(text_features.size(0), 1)
    text_features = model.encode_text_class(text, text_features, class_features)
    return text_features

def get_loss_img2text(model, img2text, images, texts, alphas, loss_img, loss_txt, args, memory=None):
    with torch.no_grad():
        image_features, _ = model.visual(images, alphas, return_attn=True)
    token_features = img2text(image_features) #qformer를 위해 unsqueeze 추가!!
    text_features = get_text_features(model, token_features, args)  # default option
    # text_features = get_text_features_add_caption(model, token_features, texts, args)
    # text_features = get_text_features_alpha(model, texts, token_features, args)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)   
    logit_scale = model.logit_scale.exp()
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        # Image loss.
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss


def get_loss_img2text_features(model, img2text, image_features, noun, caption, loss_img, loss_txt, args, data_identifier=-1):
    noun_features = get_text_features_only_caption(model, noun, args)
    noun_features += 1.0 * torch.rand(noun_features.shape[0], device=noun_features.device).unsqueeze(-1) * torch.randn(noun_features.shape, device=noun_features.device)
    image_features = torch.add(image_features, noun_features)
    token_features = img2text(image_features.unsqueeze(1))
    text_features = get_text_features(model, token_features, args)  # default option
    # text_features = get_text_features_add_caption(model, token_features, noun, args)
    # text_features = get_text_features_alpha(model, texts, token_features, args)
    # text_features = get_text_class_features(model, token_text_features, token_class_features, args)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)   
    logit_scale = model.logit_scale.exp()
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        # Image loss.
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss

# def get_loss_img2text_features(model, img2text, text2text, image_features, noun, caption, loss_img, loss_txt, args, data_identifier=-1):
#     token_features = img2text(image_features.unsqueeze(1))
#     text_features = get_text_features(model, token_features, args)  # default option
#     text_features = text2text(text_features)
#     real_text_features = get_text_features_only_caption(model, noun, args)

#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     real_text_features = real_text_features / real_text_features.norm(dim=-1, keepdim=True)
#     text_features = text_features / text_features.norm(dim=-1, keepdim=True)   
#     logit_scale = model.logit_scale.exp()
#     logit_scale = logit_scale.mean()

#     # 손실 가중치 설정 (추후 args에서 받아올 수 있도록 수정 가능)
#     text_text_weight = 1.3  # text-text contrastive loss 가중치
#     text_image_weight = 0.7 # text-image contrastive loss 가중치

#     if args.distributed and args.aggregate:
#         world_size = dist.get_world_size()
#         rank = dist.get_rank()

#         # We gather tensors from all gpus to get more negatives to contrast with.
#         gathered_real_text_features = [
#             torch.zeros_like(real_text_features) for _ in range(world_size)
#         ]
#         gathered_text_features = [
#             torch.zeros_like(text_features) for _ in range(world_size)
#         ]
#         gathered_image_features = [
#             torch.zeros_like(image_features) for _ in range(world_size)
#         ]
#         dist.all_gather(gathered_real_text_features, real_text_features)
#         dist.all_gather(gathered_text_features, text_features)
#         dist.all_gather(gathered_image_features, image_features)

#         all_real_text_features = torch.cat(
#             [real_text_features]
#             + gathered_real_text_features[:rank]
#             + gathered_real_text_features[rank + 1 :]
#         )
#         all_text_features = torch.cat(
#             [text_features]
#             + gathered_text_features[:rank]
#             + gathered_text_features[rank + 1 :]
#         )
#         all_image_features = torch.cat(
#             [image_features]
#             + gathered_image_features[:rank]
#             + gathered_image_features[rank + 1 :]
#         )

#         ground_truth = torch.arange(len(all_real_text_features)).long()
#         if args.gpu is not None:
#             ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

#         # Text-Text Contrastive Loss
#         logits_text_text = logit_scale * all_real_text_features @ all_text_features.t()
#         loss_text_text = (loss_img(logits_text_text, ground_truth) + loss_txt(logits_text_text.t(), ground_truth)) / 2

#         # Text-Image Contrastive Loss
#         logits_text_image = logit_scale * all_text_features @ all_image_features.t()
#         loss_text_image = (loss_img(logits_text_image, ground_truth) + loss_txt(logits_text_image.t(), ground_truth)) / 2

#     else:
#         ground_truth = torch.arange(len(real_text_features)).long()
#         if args.gpu is not None:
#             ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

#         # Text-Text Contrastive Loss
#         logits_text_text = logit_scale * real_text_features @ text_features.t()
#         loss_text_text = (loss_img(logits_text_text, ground_truth) + loss_txt(logits_text_text.t(), ground_truth)) / 2

#         # Text-Image Contrastive Loss  
#         logits_text_image = logit_scale * text_features @ image_features.t()
#         loss_text_image = (loss_img(logits_text_image, ground_truth) + loss_txt(logits_text_image.t(), ground_truth)) / 2

#     # 가중치를 적용한 최종 손실 계산
#     total_loss = text_text_weight * loss_text_text + text_image_weight * loss_text_image
#     return total_loss, loss_text_text, loss_text_image

# def get_loss_img2class_features(model, img2text, img2class, image_features, noun, caption, loss_class, loss_noun, args, data_identifier=-1):
#     token_text_features = img2text(image_features)
#     token_class_features = img2class(image_features) #qformer를 위해 unsqueeze 추가!!
#     # text_features = get_text_features(model, token_text_features, args)  # default option
#     # text_features = get_text_features_add_caption(model, token_features, texts, args)
#     # text_features = get_text_features_alpha(model, texts, token_features, args)
#     text_features = get_text_class_features(model, token_text_features, token_class_features, args)
#     noun_features = get_text_features_only_caption(model, noun, args)
#     text_features = text_features / text_features.norm(dim=-1, keepdim=True)   
#     noun_features = noun_features / noun_features.norm(dim=-1, keepdim=True)
#     logit_scale = model.logit_scale.exp()
#     logit_scale = logit_scale.mean()
#     if args.distributed and args.aggregate:
#         world_size = dist.get_world_size()
#         rank = dist.get_rank()

#         # We gather tensors from all gpus to get more negatives to contrast with.
#         gathered_text_features = [
#             torch.zeros_like(text_features) for _ in range(world_size)
#         ]
#         gathered_noun_features = [
#             torch.zeros_like(noun_features) for _ in range(world_size)
#         ]
#         dist.all_gather(gathered_text_features, text_features)
#         dist.all_gather(gathered_noun_features, noun_features)

#         all_text_features = torch.cat(
#             [text_features]
#             + gathered_text_features[:rank]
#             + gathered_text_features[rank + 1 :]
#         )
#         all_noun_features = torch.cat(
#             [noun_features]
#             + gathered_noun_features[:rank]
#             + gathered_noun_features[rank + 1 :]
#         )

#         ground_truth = torch.arange(len(all_noun_features)).long()
#         if args.gpu is not None:
#             ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

#         # this is needed to send gradients back everywhere.
#         # Image loss.
#         logits_per_noun = logit_scale * all_noun_features @ all_text_features.t()
#         loss_noun_val = loss_noun(logits_per_noun, ground_truth)
#         logits_per_text = logits_per_noun.t()
#         loss_txt_val = loss_class(logits_per_text, ground_truth)
#     else:
#         ground_truth = torch.arange(len(noun_features)).long()
#         if args.gpu is not None:
#             ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
#         # Image loss.
#         logits_per_noun = logit_scale * noun_features @ text_features.t()
#         loss_noun_val = loss_noun(logits_per_noun, ground_truth)
#         logits_per_text = logit_scale * text_features @ noun_features.t()
#         loss_txt_val = loss_class(logits_per_text, ground_truth)
#     total_loss = (loss_noun_val + loss_txt_val) / 2
#     return total_loss


def get_loss_img2class_features(model, img2text, img2class, image_features, noun, caption, loss_txt_class, args, data_identifier=-1):
    token_text_features = img2text(image_features)
    token_class_features = img2class(image_features)
    text_features = get_text_class_features(model, token_text_features, token_class_features, args)
    noun_features = get_text_features_only_caption(model, noun, args)
    
    # 정규화 (선택적 - MSE를 사용할 때는 제거할 수도 있음)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)   
    noun_features = noun_features / noun_features.norm(dim=-1, keepdim=True)
    
    # MSE loss 계산
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 모든 GPU에서 특징 수집
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        gathered_noun_features = [torch.zeros_like(noun_features) for _ in range(world_size)]
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gathered_noun_features, noun_features)

        all_text_features = torch.cat(
            [text_features] + gathered_text_features[:rank] + gathered_text_features[rank + 1 :]
        )
        all_noun_features = torch.cat(
            [noun_features] + gathered_noun_features[:rank] + gathered_noun_features[rank + 1 :]
        )

        # MSE loss 계산
        loss = loss_txt_class(all_text_features, all_noun_features)
    else:
        # 단일 GPU에서의 MSE loss 계산
        loss = loss_txt_class(text_features, noun_features)

    return loss


def train(model, img2text, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.eval()
    dataloader, sampler = data['train'].dataloader,  data['train'].sampler
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        if batch is None:
            continue

        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images, texts, alphas = batch[0], batch[1], batch[2]
        if len(batch) == 3 and args.use_debiased_sampler:
            data_identifier = torch.unique(batch[2])[0].numpy()
        else:
            data_identifier = -1
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            alphas = alphas.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = get_loss_img2text(m, img2text, images, texts, alphas, loss_img, loss_txt, args, data_identifier)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss_img2text(m, img2text, images, texts, alphas, loss_img, loss_txt, args, data_identifier)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        #m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})

def train_features(model, img2text, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.eval()
    dataloader, sampler = data['train'].dataloader,  data['train'].sampler
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        if batch is None:
            continue

        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        image_features, noun, caption = batch[0], batch[1], batch[2]
        if len(batch) == 3 and args.use_debiased_sampler:
            data_identifier = torch.unique(batch[2])[0].numpy()
        else:
            data_identifier = -1
        if args.gpu is not None:
            image_features = image_features.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                loss_text = get_loss_img2text_features(m, img2text, image_features, noun, caption, loss_img, loss_txt, args, data_identifier)
                scaler.scale(loss_text).backward()
                scaler.step(optimizer)

            scaler.update()

        else:
            loss_text = get_loss_img2text_features(m, img2text, image_features, noun, caption, loss_img, loss_txt, args, data_identifier)
            loss_text.backward()
            optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(image_features) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss Text: {loss_text.item():.6f}\t"
                f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR Text: {optimizer.param_groups[0]['lr']:5f}"
            )

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss_text": loss_text.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "lr_text": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})

# def train_features(model, img2text, text2text, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
#     os.environ["WDS_EPOCH"] = str(epoch)
#     model.eval()
#     dataloader, sampler = data['train'].dataloader,  data['train'].sampler
#     loss_img = nn.CrossEntropyLoss()
#     loss_txt = nn.CrossEntropyLoss()
#     if args.gpu is not None:
#         loss_img = loss_img.cuda(args.gpu)
#         loss_txt = loss_txt.cuda(args.gpu)
#     if args.distributed and sampler is not None:
#         sampler.set_epoch(epoch)

#     num_batches_per_epoch = dataloader.num_batches

#     end = time.time()
#     for i, batch in enumerate(dataloader):
#         if batch is None:
#             continue

#         step = num_batches_per_epoch * epoch + i
#         scheduler(step)

#         optimizer.zero_grad()

#         image_features, noun, caption = batch[0], batch[1], batch[2]
#         if len(batch) == 3 and args.use_debiased_sampler:
#             data_identifier = torch.unique(batch[2])[0].numpy()
#         else:
#             data_identifier = -1
#         if args.gpu is not None:
#             image_features = image_features.cuda(args.gpu, non_blocking=True)

#         data_time = time.time() - end

#         m = model.module if args.distributed or args.dp else model

#         # with automatic mixed precision.
#         if args.precision == "amp":
#             with autocast():
#                 total_loss, loss_text_text, loss_text_image = get_loss_img2text_features(m, img2text, text2text, image_features, noun, caption, loss_img, loss_txt, args, data_identifier)
#                 scaler.scale(total_loss).backward()
#                 scaler.step(optimizer)

#             scaler.update()

#         else:
#             total_loss, loss_text_text, loss_text_image = get_loss_img2text_features(m, img2text, text2text, image_features, noun, caption, loss_img, loss_txt, args, data_identifier)
#             total_loss.backward()
#             optimizer.step()

#         batch_time = time.time() - end
#         end = time.time()

#         if is_master(args) and (i % 100) == 0:
#             num_samples = i * len(image_features) * args.world_size
#             samples_per_epoch = dataloader.num_samples
#             percent_complete = 100.0 * i / num_batches_per_epoch
#             logging.info(
#                 f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
#                 f"Loss Text: {loss_text_text.item():.6f}\t"
#                 f"Loss Image: {loss_text_image.item():.6f}\t"
#                 f"Loss Total: {total_loss.item():.6f}\t"
#                 f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
#                 f"\tLR Text: {optimizer.param_groups[0]['lr']:5f}"
#             )

#             timestep = epoch * num_batches_per_epoch + i
#             log_data = {
#                 "loss_text_text": loss_text_text.item(),
#                 "loss_text_image": loss_text_image.item(),
#                 "loss_total": total_loss.item(),
#                 "data_time": data_time,
#                 "batch_time": batch_time,
#                 "lr_text": optimizer.param_groups[0]["lr"]
#             }

#             for name, val in log_data.items():
#                 name = "train/" + name
#                 if tb_writer is not None:
#                     tb_writer.add_scalar(name, val, timestep)
#                 if args.wandb:
#                     wandb.log({name: val, 'step': timestep})

# def train_features(model, img2text, img2class, data, epoch, optimizer_text, optimizer_class, scaler_text, scaler_class, scheduler_text, scheduler_class, args, tb_writer=None):
#     os.environ["WDS_EPOCH"] = str(epoch)
#     model.eval()
#     dataloader, sampler = data['train'].dataloader,  data['train'].sampler
#     loss_img = nn.CrossEntropyLoss()
#     loss_txt = nn.CrossEntropyLoss()
#     # loss_txt_class = nn.CrossEntropyLoss()
#     # loss_noun = nn.CrossEntropyLoss()
#     loss_txt_class = nn.MSELoss()
#     if args.gpu is not None:
#         loss_img = loss_img.cuda(args.gpu)
#         loss_txt = loss_txt.cuda(args.gpu)
#         # loss_txt_class = loss_txt_class.cuda(args.gpu)
#         # loss_noun = loss_noun.cuda(args.gpu)
#         loss_txt_class = loss_txt_class.cuda(args.gpu)
#     if args.distributed and sampler is not None:
#         sampler.set_epoch(epoch)

#     num_batches_per_epoch = dataloader.num_batches

#     end = time.time()
#     for i, batch in enumerate(dataloader):
#         if batch is None:
#             continue

#         step = num_batches_per_epoch * epoch + i
#         scheduler_text(step)
#         scheduler_class(step)

#         optimizer_text.zero_grad()
#         optimizer_class.zero_grad()

#         image_features, noun, caption = batch[0], batch[1], batch[2]
#         if len(batch) == 3 and args.use_debiased_sampler:
#             data_identifier = torch.unique(batch[2])[0].numpy()
#         else:
#             data_identifier = -1
#         if args.gpu is not None:
#             image_features = image_features.cuda(args.gpu, non_blocking=True)

#         data_time = time.time() - end

#         m = model.module if args.distributed or args.dp else model

#         # with automatic mixed precision.
#         if args.precision == "amp":
#             with autocast():
#                 # img2text 학습
#                 loss_text = get_loss_img2text_features(m, img2text, img2class, image_features, noun, caption, loss_img, loss_txt, args, data_identifier)
#                 scaler_text.scale(loss_text).backward()
#                 scaler_text.step(optimizer_text)
                
#                 # img2class 학습
#                 loss_class = get_loss_img2class_features(m, img2text, img2class, image_features, noun, caption, loss_txt_class, args, data_identifier)
#                 scaler_class.scale(loss_class).backward()
#                 scaler_class.step(optimizer_class)
#             scaler_text.update()
#             scaler_class.update()

#         else:
#             # img2text 학습
#             loss_text = get_loss_img2text_features(m, img2text, img2class, image_features, noun, caption, loss_img, loss_txt, args, data_identifier)
#             loss_text.backward()
#             optimizer_text.step()
            
#             # img2class 학습
#             loss_class = get_loss_img2class_features(m, img2text, img2class, image_features, noun, caption, loss_txt_class, args, data_identifier)
#             loss_class.backward()
#             optimizer_class.step()

#         batch_time = time.time() - end
#         end = time.time()

#         if is_master(args) and (i % 100) == 0:
#             num_samples = i * len(image_features) * args.world_size
#             samples_per_epoch = dataloader.num_samples
#             percent_complete = 100.0 * i / num_batches_per_epoch
#             logging.info(
#                 f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
#                 f"Loss Text: {loss_text.item():.6f}\tLoss Class: {loss_class.item():.6f}\t"
#                 f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
#                 f"\tLR Text: {optimizer_text.param_groups[0]['lr']:5f}\tLR Class: {optimizer_class.param_groups[0]['lr']:5f}"
#             )

#             timestep = epoch * num_batches_per_epoch + i
#             log_data = {
#                 "loss_text": loss_text.item(),
#                 "loss_class": loss_class.item(),
#                 "data_time": data_time,
#                 "batch_time": batch_time,
#                 "lr_text": optimizer_text.param_groups[0]["lr"],
#                 "lr_class": optimizer_class.param_groups[0]["lr"]
#             }

#             for name, val in log_data.items():
#                 name = "train/" + name
#                 if tb_writer is not None:
#                     tb_writer.add_scalar(name, val, timestep)
#                 if args.wandb:
#                     wandb.log({name: val, 'step': timestep})