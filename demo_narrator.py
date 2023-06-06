# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import urllib.request
from collections import OrderedDict

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import decord

from benchmarks.zero_shot_error_detection.lavila.lavila.data.video_transforms import Permute
from benchmarks.zero_shot_error_detection.lavila.lavila.data.datasets import get_frame_ids, video_loader_by_frames
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from lavila.models.tokenizer import MyGPT2Tokenizer
from eval_narrator import decode_one


def main(args):
    ckpt_name = 'vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth'
    ckpt_path = os.path.join('modelzoo/', ckpt_name)
    os.makedirs('modelzoo/', exist_ok=True)
    if not os.path.exists(ckpt_path):
        print('downloading model to {}'.format(ckpt_path))
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}'.format(ckpt_name),
                                   ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # instantiate the model, and load the pre-trained weights
    model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
        text_use_cls_token=False,
        project_embed_dim=256,
        gated_xattn=True,
        timesformer_gated_xattn=False,
        freeze_lm_vclm=False,  # we use model.eval() anyway
        freeze_visual_vclm=False,  # we use model.eval() anyway
        num_frames=4,
        drop_path_rate=0.
    )
    model.load_state_dict(state_dict, strict=True)
    if args.cuda:
        model.cuda()
    model.eval()

    # transforms on input frames
    crop_size = 336
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001],
                                        std=[68.5005327, 66.6321579, 70.32316305])
    ])

    video_path = os.path.join(args.input_directory_path, f'{args.recording_id}_360p.mp4')
    if os.path.exists(video_path):
        print(f'video path: {video_path} exists')
    else:
        print(f'video path: {video_path} does not exist')
        video_path = os.path.join(args.input_directory_path, f'{args.recording_id}_360p.MP4')
    narrations = []
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    num_segments = 4
    num_return_sequences = 10
    total_iterations = total_frames // 60

    os.makedirs(args.output_directory_path, exist_ok=True)
    output_path = os.path.join(args.output_directory_path, f'{args.recording_id}.txt')
    for iteration in range(total_iterations):
        print('iteration {}/{}'.format(iteration, total_iterations))
        frame_ids = get_frame_ids(iteration * 60, min((iteration + 1) * 60, total_frames), num_segments=num_segments,
                                  jitter=False)
        frames = video_loader_by_frames('./', video_path, frame_ids)

        frames = val_transform(frames)
        frames = frames.unsqueeze(0)  # fake a batch dimension

        tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
        with torch.no_grad():
            if args.cuda:
                frames = frames.cuda(non_blocking=True)
            image_features = model.encode_image(frames)
            generated_text_ids, ppls = model.generate(
                image_features,
                tokenizer,
                target=None,  # free-form generation
                max_text_length=77,
                top_k=None,
                top_p=0.95,  # nucleus sampling
                num_return_sequences=num_return_sequences,  # number of candidates: 10
                temperature=0.7,
                early_stopping=True,
            )

        print('-----------------')
        for i in range(num_return_sequences):
            generated_text_str = decode_one(generated_text_ids[i], tokenizer)
            narrations.append(generated_text_str)
            print('{}: {}'.format(i, generated_text_str))

    with open(output_path, 'w') as f:
        for narration in narrations:
            f.write(narration + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila narrator demo')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--input_directory_path', default='/data/ANNOTATION', type=str, help='input directory path')
    parser.add_argument('--recording_id', default='12_5', type=str, help='input video path')
    parser.add_argument('--output_directory_path', default='narrations.txt', type=str, help='output path')
    args = parser.parse_args()
    main(args)
