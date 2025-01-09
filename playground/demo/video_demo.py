import argparse
import torch

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

from transformers import AutoConfig

import cv2
import base64

import numpy as np
import warnings

# Enable stack traces for all warnings
#warnings.simplefilter("always")  # Ensure all warnings are shown
#warnings.filterwarnings("always")  # Force warnings to display

# Optionally hook into `warnings.showwarning`
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    import traceback
    print(f"Warning: {message} (Category: {category})")
    traceback.print_stack()

#warnings.showwarning = custom_showwarning

# Run your code


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="Path to the video files.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str)
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

def run_inference(args):
    """Run inference on ActivityNet QA DataSet using the Video-ChatGPT model."""

    model_name = get_model_name_from_path(args.model_path)
    # Set model configuration parameters if they exist
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_newline_position"] = args.mm_newline_position

        cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

        if "qwen" not in args.model_path.lower():
            if "224" in cfg_pretrained.mm_vision_tower:
                # suppose the length of text tokens is around 1000, from bo's report
                least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
            else:
                least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

            scaling_factor = math.ceil(least_token_number/4096)
            if scaling_factor >= 2:
                if "vicuna" in cfg_pretrained._name_or_path.lower():
                    print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config, attn_implementation='sdpa')

    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False

    os.makedirs(args.output_dir, exist_ok=True)

    json_out_fp = os.path.join(args.output_dir, f"{args.output_name}.json")
    #ans_file = open(answers_file, "w")

    all_video_paths = []

    if os.path.isdir(args.video_path):
        for filename in os.listdir(args.video_path):
            cur_video_path = os.path.join(args.video_path, f"{filename}")
            all_video_paths.append(os.path.join(args.video_path, cur_video_path))
    else:
        all_video_paths.append(args.video_path)

    #for args.video_path in all_video_pathes:
    for video_path in all_video_paths:

        sample_set = {}
        #question = args.prompt
        #question = "what are the key events in this video?"
        #sample_set["Q"] = question
        #sample_set["video_name"] = args.video_path

        assert os.path.exists(video_path)
        video,frame_time,video_time = load_video(video_path, args)
        # load splittimes, split into scenes, loop through and write output to
        # a file with scene num appended
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
        video = [video]
        #qs = question
        qs = "what are the key events in this video?"
        if args.add_time_instruction:
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
            qs = f'{time_instruciton}\n{qs}'
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        #conv = ''
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        #prompt = qs

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        if tokenizer.pad_token_id is None:
            if "qwen" in tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        #stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        #keywords = [stop_str]
        #stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        if args.cpu:
            model = model.float().cpu()
            attention_masks = attention_masks.cpu()
            video = [video[0].cpu().float()]
            input_ids = input_ids.cpu()
        else:
            model = model.cuda()
        with torch.inference_mode():
            if "mistral" not in cfg_pretrained._name_or_path.lower():
                output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1,num_beams=1,use_cache=True)
            else:
                output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1, use_cache=True)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"Question: {prompt}\n")
        print(f"Response: {outputs}\n")

        #if "mistral" not in cfg_pretrained._name_or_path.lower():
            #if outputs.endswith(stop_str):
                #outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()

        sample_set[video_path] = outputs
        out_fn = os.path.basename(video_path).split('.')[0] + 'txt'
        os.makedirs('outputs', exist_ok=True)
        with open(f'outputs/{out_fn}', 'w') as f:
            f.write(outputs)
        #ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        #ans_file.flush()

    with open(json_out_fp, 'w') as f:
        json.dump(sample_set, f)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
