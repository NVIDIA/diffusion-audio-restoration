# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os

# # If there is Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
# os.environ["MKL_THREADING_LAYER"] = "GNU"
# import numpy as np
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import numpy as np 
import json
import argparse
import glob
from subprocess import Popen, PIPE
import yaml
import time 
from datetime import datetime
import shutil
import csv
from tqdm import tqdm
from copy import deepcopy

import librosa
import soundfile as sf


def read_standard_csv(root_folder, filename):
    all_files = {
        "train": [],
        "validation": [],
        "test": []
    }

    with open(os.path.join(root_folder, filename)) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            assert len(row) == 3
            split, audio_filename, duration = row
            split = split.strip()
            audio_filename = audio_filename.strip()
            duration = float(duration)
            sample_rate = None  # 44100  # not used
            all_files[split].append((audio_filename, duration, sample_rate))

    return all_files


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def save_yaml(data, prefix="../configs/temp"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd_num = np.random.rand()
    rnd_num = rnd_num - rnd_num % 0.000001
    file_name = f"{prefix}_{timestamp}_{rnd_num}.yaml"
    with open(file_name, 'w') as f:
        yaml.dump(data, f)
    return file_name


def shell_run_cmd(cmd):
    print('running:', cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = p.communicate()
    print(stdout)
    print(stderr)


def inpaint_one_sample(dataset_name, audio_filename, exp_root, exp_name, inpaint_length, inpaint_every, max_segment_length=-1, custom_output_subdir=None):
    assert 0 < inpaint_length < inpaint_every < 10 

    # get paths ready
    if custom_output_subdir is not None:
        output_subdir = custom_output_subdir
    else:
        output_subdir = '_'.join(audio_filename.split('/')[-3:])  # get reasonably short filename
        output_subdir = '.'.join(output_subdir.split('.')[:-1])  # remove suffix

    output_dir = os.path.join(exp_root, exp_name, dataset_name, 'inpaint_{}_every_{}'.format(inpaint_length, inpaint_every))
    if os.path.exists(os.path.join(output_dir, output_subdir, 'recon.wav')):
        print(audio_filename, ' - already inpainted')
        return
    # elif custom_output_subdir is None:
    #     shutil.rmtree(os.path.join(output_dir, output_subdir))
    
    # copy true file
    os.makedirs(os.path.join(output_dir, output_subdir), exist_ok=True)
    audio_suffix = audio_filename.split('.')[-1]
    original_target = os.path.join(output_dir, output_subdir, 'original.{}'.format(audio_suffix))
    if not os.path.exists(original_target):
        shutil.copy(audio_filename, original_target)
    orig_audio, orig_sr = librosa.load(audio_filename, sr=None)
    duration = len(orig_audio) / orig_sr

    if (custom_output_subdir is None) and max_segment_length > 0 and duration > max_segment_length + 2.0:
        # split original audio into segments
        assert max_segment_length % inpaint_every == 0, 'max_segment_length={}, inpaint_every={}'.format(max_segment_length, inpaint_every)
        n_segments = int(np.ceil(duration / max_segment_length))
        print('duration is {:.2f}; segment into {} parts'.format(duration, n_segments))
        for i in range(n_segments):
            orig_segment = orig_audio[int(i*max_segment_length*orig_sr):int((i+1)*max_segment_length*orig_sr)]
            part_dir = os.path.join(output_dir, output_subdir, 'tmp_part{}'.format(i))
            part_audio_filename = os.path.join(part_dir, 'original.{}'.format(audio_suffix))
            os.makedirs(part_dir, exist_ok=True)
            sf.write(part_audio_filename, orig_segment, orig_sr)
            inpaint_one_sample(dataset_name, part_audio_filename, exp_root, exp_name, inpaint_length, inpaint_every, custom_output_subdir=part_dir)
        
        # concatenate all 
        recon_audio = None
        for i in range(n_segments):
            part_dir = os.path.join(output_dir, output_subdir, 'tmp_part{}'.format(i))
            recon_audio_part, recon_sr = librosa.load(os.path.join(part_dir, 'recon.wav'), sr=None)
            if recon_audio is None:
                recon_audio = recon_audio_part
            else:
                recon_audio = np.append(recon_audio, recon_audio_part)
        sf.write(os.path.join(output_dir, output_subdir, 'recon.wav'), recon_audio, recon_sr)
    
    else:
        pass

    # Load, modify, and store yaml file for the specific file
    template_yaml_file = '../configs/inference_files_inpainting.yaml'
    inference_config = load_yaml(template_yaml_file)
    inference_config['data']['predict_filelist'] = [{
        'filepath': audio_filename,
        'output_subdir': output_subdir
    }]

    _each_transforms_aug = deepcopy(inference_config['data']['transforms_aug'][0])
    inference_config['data']['transforms_aug'] = []
    
    starts, ends = [], []
    for i in range(int(duration // inpaint_every)):
        start = i * inpaint_every + (inpaint_every - inpaint_length) / 2
        end = i * inpaint_every + (inpaint_every + inpaint_length) / 2
        _each_transforms_aug['init_args']['start_time'] = start
        _each_transforms_aug['init_args']['end_time'] = end

        inference_config['data']['transforms_aug'].append(deepcopy(_each_transforms_aug))
        starts.append(start)
        ends.append(end)

    temporary_yaml_file = save_yaml(inference_config)

    # compute degraded audio
    degraded_audio = deepcopy(orig_audio)
    for t1, t2 in zip(starts, ends):
        degraded_audio[int(t1*orig_sr):int(t2*orig_sr)] = 0
    sf.write(os.path.join(output_dir, output_subdir, 'degraded.{}'.format(audio_suffix)), degraded_audio, orig_sr)

    # run inpainting command
    cmd = "cd ../; \
        python ensembled_inference.py predict \
            -c configs/{}.yaml \
            -c {} \
            --model.fast_inpaint_mode=true \
            --model.predict_n_steps=200 \
            --model.predict_output_dir={}; \
        cd inference/".format(exp_name, temporary_yaml_file.replace('../', ''), output_dir)
    
    shell_run_cmd(cmd)
    
    os.remove(temporary_yaml_file)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-dn','--dataset_name', help='dataset name', required=True)
    parser.add_argument('-exp','--exp_name', help='exp_name', required=True)
    parser.add_argument('-inp_len','--inpaint_length', type=float, help='inpaint_length', required=True)
    parser.add_argument('-inp_every','--inpaint_every', type=float, help='inpaint_every', required=True)
    parser.add_argument('-seg_len','--max_segment_length', type=float, default=-1, help='maximum segment length for inpainting')
    parser.add_argument('-start','--start', type=int, help='start', default=0)
    parser.add_argument('-end','--end', type=int, help='end', default=-1)
    args = parser.parse_args()

    manifest_root_folder = 'PATH/TO/MANIFEST/FOLDER'
    exp_root = './exp'

    dataset_name = args.dataset_name
    manifest_filename = '{}_manifest.csv'.format(dataset_name)
    all_files = read_standard_csv(manifest_root_folder, manifest_filename)

    exp_name = args.exp_name
    inpaint_length = args.inpaint_length
    inpaint_every = args.inpaint_every
    max_segment_length = args.max_segment_length

    start = args.start
    end = args.end if args.end > args.start else len(all_files['test'])

    for row in tqdm(all_files['test'][start:end]):
        (audio_filename, duration, sample_rate) = row
        inpaint_one_sample(dataset_name, audio_filename, exp_root, exp_name, inpaint_length, inpaint_every, max_segment_length)


if __name__ == '__main__':
    main()

