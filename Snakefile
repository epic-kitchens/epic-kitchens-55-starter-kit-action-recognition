from multiprocessing import cpu_count
import os
import pandas as pd
import numpy as np

configfile: 'config.yml'
MAX_THREADS = cpu_count()

def read_video_id_csv(path):
    return pd.read_csv(path, index_col='video_id')

def video_id_to_participant_id(video_id):
    assert len(video_id) == len('PXX_YY')
    return video_id[:len('PXX')]

TRAIN_VIDEOS = read_video_id_csv('train_video_ids.csv')
TEST_SEEN_VIDEOS = read_video_id_csv('test_seen_video_ids.csv')
TEST_UNSEEN_VIDEOS = read_video_id_csv('test_unseen_video_ids.csv')
TEST_VIDEOS = pd.concat([TEST_SEEN_VIDEOS, TEST_UNSEEN_VIDEOS])
ALL_VIDEOS = pd.concat([TRAIN_VIDEOS, TEST_VIDEOS])


ALL_IDS = ALL_VIDEOS.index.values
TRAIN_IDS = TRAIN_VIDEOS.index.values
TEST_IDS = TEST_VIDEOS.index.values
TEST_SEEN_IDS = TEST_SEEN_VIDEOS.index.values
TEST_UNSEEN_IDS = TEST_UNSEEN_VIDEOS.index.values
IDS = {
    'train': TRAIN_IDS,
    'test_seen': TEST_SEEN_IDS,
    'test_unseen': TEST_UNSEEN_IDS,
}



def regex_options(iterable):
    return "(" + "|".join(iterable) + ")"


wildcard_constraints:
    participant_id=regex_options(set(map(video_id_to_participant_id, ALL_IDS))),
    video_id=regex_options(ALL_IDS),
    train_video_id=regex_options(set(ALL_IDS) - set(TEST_IDS)),
    test_video_id=regex_options(set(TEST_IDS)),
    test_seen_video_id=regex_options(set(TEST_SEEN_IDS)),
    test_unseen_video_id=regex_options(set(TEST_UNSEEN_IDS)),
    set="(train|test_seen|test_unseen)",
    modality="(flow|rgb)"


### DOWNLOAD METADATA ###

rule download_verb_classes:
    output: "data/raw/EPIC_verb_classes.csv"
    shell:
        "wget https://github.com/epic-kitchens/annotations/raw/master/EPIC_verb_classes.csv -O {output}"


rule download_noun_classes:
    output: "data/raw/EPIC_noun_classes.csv"
    shell:
        "wget https://github.com/epic-kitchens/annotations/raw/master/EPIC_noun_classes.csv -O {output}"


rule download_action_labels:
    output: "data/raw/EPIC_train_action_labels.pkl"
    shell:
        "wget https://github.com/epic-kitchens/annotations/raw/master/EPIC_train_action_labels.pkl -O {output}"

rule download_timestamps:
  output: "data/raw/EPIC_test_{split}_timestamps.pkl"
  wildcard_constraints:
      split="s(1|2)"
  shell:
    "wget https://github.com/epic-kitchens/annotations/raw/master/EPIC_test_{wildcards.split}_timestamps.pkl -O {output}"

rule download_metadata:
  input:
    rules.download_verb_classes.output,
    rules.download_noun_classes.output,
    "data/processed/train_labels.pkl",
    "data/processed/test_seen_labels.pkl",
    "data/processed/test_unseen_labels.pkl",



### DOWNLOAD MEDIA ###

rule download_train_video_frames:
    output: "data/raw/{modality}/{participant_id}/{train_video_id}.tar"
    shell:
        """
        wget \
          "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/{wildcards.modality}/train/{wildcards.participant_id}/{wildcards.train_video_id}.tar" \
          -O {output:q} || rm {output:q}
        """

rule download_test_video_frames:
    output: "data/raw/{modality}/{participant_id}/{test_video_id}.tar"
    shell:
        """
        wget \
          "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/{wildcards.modality}/test/{wildcards.participant_id}/{wildcards.test_video_id}.tar" \
          -O {output:q} || rm {output:q}
          """

rule download_all_rgb_frames:
    input: ["data/raw/rgb/{}/{}.tar".format(video_id_to_participant_id(video_id), video_id) for video_id in ALL_IDS]


rule download_all_flow_frames:
  input: ["data/raw/flow/{}/{}.tar".format(video_id_to_participant_id(video_id), video_id) for video_id in ALL_IDS]


rule download_all_frames:
  input:
    rules.download_all_rgb_frames.output,
    rules.download_all_flow_frames.output



### PROCESS MEDIA ###

rule untar_frames:
    input: "data/raw/{modality}/{participant_id}/{video_id}.tar"
    output: directory(config['tmp_dir'] + "/raw/{modality}/{participant_id}/{video_id}")
    shell: "mkdir -p {output:q}; tar -xvf {input:q} -C {output:q} > /dev/null"


rule rename_train_labels:
  input: "data/raw/EPIC_train_action_labels.pkl"
  output: "data/processed/train_labels.pkl"
  shell: 'mkdir -p "$(basename {output:q})"; cp {input:q} {output:q}'


test_set_to_split = {
  'seen': 's1',
  'unseen': 's2',
}


rule rename_test_timestamps:
  input: lambda wildcards: "data/raw/EPIC_test_" + test_set_to_split[wildcards.test_set] + "_timestamps.pkl"
  output: "data/processed/test_{test_set}_labels.pkl"
  wildcard_constraints:
      test_set="(seen|unseen)"
  shell: 'mkdir -p "$(basename {output:q})"; cp {input:q} {output:q}'


rule segment_frames_for_video:
    input:
        frames=config['tmp_dir'] + "/raw/{modality}/{participant_id}/{video_id}",
        labels="data/processed/{set}_labels.pkl",
    output: directory(config['tmp_dir'] + "/interim/{modality}_{set}_segments/{participant_id}/{video_id}")
    shell:
        """
        mkdir -p {output:q}
        python -m epic_kitchens.preprocessing.split_segments \
          $(basename {input.frames:q}) \
          {input.frames:q} \
          {output:q} \
          {input.labels:q} \
          {wildcards.modality} \
          --fps 60 \
          --frame-format 'frame_%010d.jpg' \
          --of-stride 2 \
          --of-dilation 3
        """


def segments(wildcards):
    return [config['tmp_dir'] +
        "/interim/{modality}_{set}_segments/{participant_id}/{video_id}".format(
        participant_id=video_id_to_participant_id(video_id),
        video_id=video_id,
        set=wildcards.set,
        modality=wildcards.modality
      ) for video_id in IDS[wildcards.set]]


rule gulp_train:
    input:
        segmented_frames=segments,
        labels="data/processed/{set}_labels.pkl",
    output: directory(config['tmp_dir'] + "/processed/gulp/{modality}_{set}")
    threads: MAX_THREADS
    params:
        segments_per_chunk=100,
        frame_size=-1,
    wildcard_constraints:
      set="train"
    shell:
      """
      python -m epic_kitchens.gulp \
        --num-workers {threads} \
        --frame-size {params.frame_size} \
        --segments-per-chunk {params.segments_per_chunk} \
        "$(dirname "$(dirname {input.segmented_frames[0]:q})")" \
        {output:q} \
        {input.labels:q} \
        {wildcards.modality}
      """


rule gulp_test:
    input:
        segmented_frames=segments,
        labels="data/processed/{set}_labels.pkl",
    output: directory(config['tmp_dir'] + "/processed/gulp/{modality}_{set}")
    threads: MAX_THREADS
    wildcard_constraints:
        set="test_(seen|unseen)"
    params:
        segments_per_chunk=100,
        frame_size=-1,
    shell:
      """
      python -m epic_kitchens.gulp \
        --num-workers {threads} \
        --frame-size {params.frame_size} \
        --segments-per-chunk {params.segments_per_chunk} \
        --unlabelled \
        "$(dirname "$(dirname {input.segmented_frames[0]:q})")" \
        {output:q} \
        {input.labels:q} \
        {wildcards.modality}
      """

if config['tmp_dir'] != 'data':
    rule copy_gulp_dirs:
        input: config['tmp_dir'] + "/processed/gulp/{modality}_{set}"
        output: directory("data/processed/gulp/{modality}_{set}")
        shell: "cp -r {input:q} {output:q}"



rule gulp_all:
    input: ['data/processed/gulp/' + path for path in ('rgb_test_seen', 'rgb_test_unseen', 'rgb_train', 'flow_train', 'flow_test_seen', 'flow_test_unseen')]


rule all:
    input:
        rules.gulp_all.input,
        rules.download_metadata.input,

# vim: set ft=yaml:
