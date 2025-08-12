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


def upsample_one_sample(dataset_name, audio_filename, exp_root, exp_name, cutoff_freq):
    # get paths ready
    output_subdir = '_'.join(audio_filename.split('/')[-3:])  # get reasonably short filename
    output_subdir = '.'.join(output_subdir.split('.')[:-1])  # remove suffix
    output_dir = os.path.join(exp_root, exp_name, dataset_name, 'cutoff_freq={}'.format(cutoff_freq))

    # Load, modify, and store yaml file for the specific file
    template_yaml_file = '../configs/inference_files_upsampling.yaml'
    inference_config = load_yaml(template_yaml_file)
    inference_config['data']['predict_filelist'] = [{
        'filepath': audio_filename,
        'output_subdir': output_subdir
    }]
    inference_config['data']['transforms_aug'][0]['init_args']['upsample_mask_kwargs'] = {
        'min_cutoff_freq': cutoff_freq,
        'max_cutoff_freq': cutoff_freq
    }
    temporary_yaml_file = save_yaml(inference_config)

    # copy true file
    os.makedirs(os.path.join(output_dir, output_subdir), exist_ok=True)
    shutil.copy(
        audio_filename, 
        os.path.join(output_dir, output_subdir, 'original.{}'.format(audio_filename.split('.')[-1]))
    )
    orig_audio, orig_sr = librosa.load(audio_filename, sr=None)
    degraded_audio = librosa.resample(orig_audio, orig_sr=orig_sr, target_sr=cutoff_freq*2)
    degraded_audio = librosa.resample(degraded_audio, orig_sr=cutoff_freq*2, target_sr=orig_sr)
    sf.write(os.path.join(output_dir, output_subdir, 'degraded.{}'.format(audio_filename.split('.')[-1])), degraded_audio, orig_sr)

    # run upsampling command
    if not os.path.exists(os.path.join(output_dir, output_subdir, 'recon.wav')):
        cmd = "cd ../; \
            python ensembled_inference.py predict \
                -c configs/{}.yaml \
                -c {} \
                --model.predict_output_dir={}; \
            cd inference/".format(exp_name, temporary_yaml_file.replace('../', ''), output_dir)
        shell_run_cmd(cmd)
    else:
        print(audio_filename, ' - already upsampled')
    
    os.remove(temporary_yaml_file)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-dn','--dataset_name', help='dataset name', required=True)
    parser.add_argument('-exp','--exp_name', help='exp_name', required=True)
    parser.add_argument('-cf','--cutoff_freq', type=int, help='cutoff_freq', required=True)
    parser.add_argument('-start','--start', type=int, help='start', default=0)
    parser.add_argument('-end','--end', type=int, help='end', default=-1)
    args = parser.parse_args()

    manifest_root_folder = 'PATH/TO/MANIFEST/FOLDER'
    exp_root = './exp'

    dataset_name = args.dataset_name
    manifest_filename = '{}_manifest.csv'.format(dataset_name)
    all_files = read_standard_csv(manifest_root_folder, manifest_filename)

    exp_name = args.exp_name
    cutoff_freq = args.cutoff_freq

    start = args.start
    end = args.end if args.end > args.start else len(all_files['test'])

    for row in tqdm(all_files['test'][start:end]):
        (audio_filename, duration, sample_rate) = row
        upsample_one_sample(dataset_name, audio_filename, exp_root, exp_name, cutoff_freq)


if __name__ == '__main__':
    main()

