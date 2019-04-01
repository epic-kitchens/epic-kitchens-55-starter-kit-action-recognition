# EPIC-Kitchens Action recognition starter kit
[![CircleCI status badge](https://img.shields.io/circleci/project/github/epic-kitchens/starter-kit-action-recognition/master.svg)](https://circleci.com/gh/epic-kitchens/starter-kit-action-recognition)

This is a tutorial introduction to the EPIC-kitchens egocentric action dataset.  
We'll introduce you to EPIC and some tools like Snakemake, a python build tool, for managing the process
of

  * Downloading RGB frames, Optical flow frames, and action metadata.
  * Integrating the metadata and frames into a format suitable for using a model training loop.

Steps:

1. [Get your environment setup with Conda](#environment-set-up)
2. [Download and process the dataset with Snakemake](#data-pipeline-snakemake)
3. [Explore the dataset and learn how to load frames and labels in Jupyter
   notebooks](#notebooks)


## Environment set-up

In this section we'll set up a virtual environment with all the tools and
libraries you need, allowing you to easily work with EPIC.

We provide a conda
[`environment.yaml`](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
file describing the environment you need to work with EPIC

If you don't have conda, go and [install it](https://conda.io/miniconda.html),
either miniconda or anaconda will be fine.

Download the repository, and create the conda virtual environment:

```console
$ git clone https://github.com/epic-kitchens/starter-kit-action-recognition.git epic
$ cd epic
$ conda env create -n epic -f environment.yaml
$ conda activate epic
```


## Data pipeline: Snakemake

In this section we'll download EPIC and perform some processing steps
to transform it into a format suitable for working with in a model training
loop.

We use [Snakemake](https://snakemake.readthedocs.io/en/stable/), a build tool
similar to `make`, to manage the downloading of all media and metadata,
segmentation of frames into action segments, and ingestion of data into GulpIO, a
high performance format for video storage for machine learning.

To download the EPIC action segment labels, and build a new pickled dataframe 
for the train and test sets you can run `snakemake data/processed/{test_{,un}seen,train}_labels.pkl`.

You have now built `data/processed/train_labels.pkl`,
`data/processed/test_seen_labels.pkl`, and
`data/processed/test_unseen_labels.pkl`, which contain the train and test labels
of EPIC stored as a [pickled](https://docs.python.org/3/library/pickle.html)
[dataframe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).
You can now open up a python session and inspect the labels with
[`pd.read_pickle('./data/processed/train_labels.pkl')`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_pickle.html).

You should now build the gulp directories that contain all the flow and RGB frames which
are used in the second notebook, and to be used for training purposes.

**NOTE**: *If you are running on a machine where you have separate fast storage and
  are reading from a parallel network file system, you might want to set
  the `tmp_dir` variable in `config.yml` to a path residing on that file system. This
  will instruct Snakemake where to extract the frames and where to gulp the
  files to disk.*

```
$ snakemake -p gulp_all
```

This will download the RGB (220GB) and flow (100GB) frames, segment them using
the test/train labels, and finally gulp the data. 
WARNING: The final few steps which involve gulping the frames will take a *very* long
time if you aren't using a SSD - as such, we recommend that you locate the `data` directory on a SSD and 
symlink the directory here.

You should also download the class CSVs that describe the mapping between verbs/nouns and their numerical 
classes.

```
$ snakemake -p download_metadata
```

## Notebooks

In this section we'll investigate the actions within EPIC and learn how to
use the [`epic-kitchens`](https://github.com/epic-kitchens/epic-lib) library
that provides software for common use cases. Ensure you have run `snakemake -j
$(nproc) all` before using these notebooks as they use files created from the
pipeline.

1. [Introduction to EPIC](./notebooks/1.0-intro.ipynb)
2. [Reading frames with GulpIO](./notebooks/2.0-gulp.ipynb) 


## Appendix A: Importing data for training with GulpIO

Both RGB frames and flow are encoded as JPEG files. Individual files to be slow
to read from disk so we suggest employing
[GulpIO](https://github.com/TwentyBN/GulpIO) for storing the data in a manner
suitable for training.

We typically structure the files like so:

```
epic
|--- rgb-segments
|    |--- P01
|    |    |--- P01_01
|    |    |--- ...
|    |    |--- P01_19
|    ...
|--- flow-segments
|    |--- P01
|    |    |--- P01_01
|    |    |    |--- u
|    |    |    |--- v
|    |    |--- ...
|    |    |--- P01_19
|    ...
```

Inside each of these folders are named segments in the form
`<video_id>_<uid>_<narration>` (e.g. `P01_01_0_open-door`)

```
<video_id> := regex(P\d\d_\d\d)
<uid> := regex(\d+)
<narration> := <word>[-<word>]*
<word> := [a-zA-Z][a-zA-Z0-9_]*
```


## Appendix B: Manual Download

If you don't wish to use Snakemake to download the data, you can choose
to manually download it yourself.

### Media

* [**Original videos**](https://data.bris.ac.uk/data/dataset/a87271fa45d89106e5f81e47430ab6b7)
  (1920x1080, 60 FPS, 701 GB), 
  [train download script](https://github.com/epic-kitchens/download-scripts/blob/master/videos/download_train.sh),
  [test download script](https://github.com/epic-kitchens/download-scripts/blob/master/videos/download_test.sh)
* [**RGB frames**](https://data.bris.ac.uk/data/dataset/34cc87ec9dbe769931dfd21a7ec22df2) 
  (456x256, 60 FPS, 220 GB), 
  [download script](https://github.com/epic-kitchens/download-scripts/blob/master/frames_rgb_flow/download_rgb.sh)
* [**Dense Optical flow frames**](https://github.com/epic-kitchens/download-scripts/blob/master/frames_rgb_flow/download_flow.sh)
  (456x256, TVL1, 96 GB), 
  [download script](https://github.com/epic-kitchens/download-scripts/blob/master/frames_rgb_flow/flow/train/download_P03.sh)
* [**RGB frames for object detection**](https://data.bris.ac.uk/data/dataset/b5ac5fa96e0969c1f28a41bae58771ad)
  (1920x1080, 2FPS),
  [download script](https://github.com/epic-kitchens/download-scripts/blob/master/download_object_detection_images.sh)
