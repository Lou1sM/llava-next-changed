from time import time
import av
import argparse
import os
from os.path import join

import torch
import numpy as np
torch.from_numpy(np.ones(4))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from natsort import natsorted

import cv2
import base64



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='lmms-lab/LLaVA-Video-7B-Qwen2')
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--data-dir-prefix", type=str, default='.')
    parser.add_argument("--conv-mode", type=str, default='default')
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--recompute", action='store_true')
    parser.add_argument("--no-model", action='store_true')
    parser.add_argument('--show-name', type=str, default='friends')
    parser.add_argument('--season', type=int, default=2)
    parser.add_argument('--ep', type=int, required=True)
    return parser.parse_args()

def load_video(video_path,args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()

    return spare_frames,frame_time,video_time

import subprocess
from PIL import Image
from io import BytesIO

def extract_frames_ffmpeg(video_path, timepoints):
    # Extract frames at specific timestamps
    images = []
    for t in timepoints:
        cmd = [
            'ffmpeg', '-ss', str(t), '-i', video_path,
            '-frames:v', '1', '-f', 'image2pipe',
            '-vcodec', 'png', '-'
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buffer = BytesIO(result.stdout)
        try:
            img = Image.open(buffer)
            images.append(img.copy())
            img.close()
        except:
            #images.append(None)  # Handle failed frame extraction
            pass

    return images

import ffmpeg

def vid_duration(video_path):
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    return duration

def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames

def run_inference(args, tokenizer, model, image_processor, show_name, season, episode):
    """Run inference on ActivityNet QA DataSet using the Video-ChatGPT model."""

    outputs_dict = {}
    vid_subpath = f'{show_name}/season_{season}/episode_{episode}'
    #video_path = join('../amazon_video/data/full-videos', f'{vid_subpath}.mp4')
    video_path = join(args.data_dir_prefix, 'tvqa-videos', f'{vid_subpath}.mp4')
    print('path', video_path)
    assert os.path.exists(video_path)
    scene_split_points = np.load(f'{args.data_dir_prefix}/tvqa-kfs-by-scene/{vid_subpath}/scenesplit_timepoints.npy')
    #args.for_get_frames_num *= len(scene_split_points)+1
    #video,frame_time,video_time = load_video(video_path, args)
    video_time = vid_duration(video_path)
    #ext_split_points = np.array([0] + list(scene_split_points) + [video_time])
    ext_split_points = np.array([0] + list(scene_split_points))
    all_scene_videos = []
    for i, start in enumerate(ext_split_points[:-1]):
        end = ext_split_points[i+1]
        scene_timepoints = np.linspace(start, end, 4)
        scene_frames = extract_frames_ffmpeg(video_path, scene_timepoints)
        svid = np.stack([np.array(x) for x in scene_frames])
        all_scene_videos.append(svid)
    #idx_split_points = (ext_split_points*args.for_get_frames_num / video_time).astype(int)
    #print(idx_split_points)
    #if idx_split_points.max()>1e5:
    #    breakpoint()
    #all_videos = [video[idx_split_points[i]:idx_split_points[i+1]] for i in range(len(idx_split_points)-1)]
    print('scene videos shape:', [v.shape for v in all_scene_videos])
    # load splittimes, split into scenes, loop through and write output to
    # a file with scene num appended
    os.makedirs(out_dir:=f'{args.data_dir_prefix}/lava-outputs/{vid_subpath}', exist_ok=True)
    json_out_fp = os.path.join(out_dir, 'all.json')
    # Set model configuration parameters if they exist

    if args.no_model:
        return 0,0
    run_starttime = time()
    for i, scene_video in enumerate(all_scene_videos):
        out_fp = join(out_dir, f'scene{i}')
        print(f'scene{i} vid shape:', scene_video.shape)
        if scene_video.shape[0] > 4:
            idxs = torch.linspace(0, len(scene_video)-1, 4).int()
            scene_video = scene_video[idxs]
            print('reshaping to:', scene_video.shape)
        assert len(scene_video) <= 4
        if len(scene_video)==0:
            continue
        if os.path.exists(out_fp) and not args.recompute:
            print(f'{out_fp} already exists, skipping')
            continue
        scene_video = image_processor.preprocess(scene_video, return_tensors="pt")["pixel_values"].half().cuda()
        #scene_video = image_processor.preprocess(scene_video, return_tensors=None)
        scene_video = [scene_video]

        if getattr(model.config, "force_sample", None) is not None:
            args.force_sample = model.config.force_sample
        else:
            args.force_sample = False

        if getattr(model.config, "add_time_instruction", None) is not None:
            args.add_time_instruction = model.config.add_time_instruction
        else:
            args.add_time_instruction = False

        qs = f"what are the specific plot points in this scene of the TV show {show_name}?"
        #if args.add_time_instruction:
            #time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
            #qs = f'{time_instruciton}\n{qs}'
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        if tokenizer.pad_token_id is None:
            if "qwen" in tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        if args.cpu:
            model = model.float().cpu()
            attention_masks = attention_masks.cpu()
            scene_video = [scene_video[0].cpu().float()]
            input_ids = input_ids.cpu()
        else:
            model = model.cuda()
        with torch.inference_mode():
            output_ids = model.generate(inputs=input_ids, images=scene_video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=60, top_p=0.1,num_beams=1,use_cache=True)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        #print(f"Question: {prompt}\n")
        print(f"Response: {outputs}\n")

        outputs = outputs.strip()

        assert f'scene{i}' not in outputs_dict
        outputs_dict[f'scene{i}'] = outputs
        print('saving to', out_fp)
        with open(out_fp, 'w') as f:
            f.write(outputs)

    runtime = time()-run_starttime
    print(f'run time: {runtime:.3f}, per scene: {runtime/len(all_videos):.3f}')
    with open(json_out_fp, 'w') as f:
        json.dump(outputs_dict, f)


if __name__ == "__main__":
    args = parse_args()
    model_name = get_model_name_from_path(args.model_path)
    overwrite_config = {}
    overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
    overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
    overwrite_config["mm_newline_position"] = args.mm_newline_position
    load_starttime = time()
    if args.no_model:
        tokenizer, model, image_processor = None, None, None
    else:
        tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config, attn_implementation='sdpa')
    print(f'load time: {time()-load_starttime:.3f}')
    seaseps = []
    show_data_dir = join(args.data_dir_prefix, 'tvqa-videos', args.show_name)
    if args.season == -1:
        seass_to_compute = natsorted([fn[7:] for fn in os.listdir(show_data_dir)])
    else:
        seass_to_compute = [args.season]

    for seas in seass_to_compute:
        if args.ep == -1:
            for fn in natsorted(os.listdir(f'{show_data_dir}/season_{seas}')):
                ep_num = fn[8:].removesuffix('.mp4')
                seaseps.append((seas, ep_num))
        else:
            seaseps.append((seas, args.ep))

    print(seaseps)
    for seas, ep in seaseps:
        run_inference(args, tokenizer, model, image_processor, args.show_name, seas, ep)
