#! /bin/zsh

set -e
dir=$(dirname $(readlink -fn --  $0))
cd $dir/..

module_name=$(basename $dir)
experiment_dir=$dir/experiments/$1
dataset_dir=$dir/datasets/$2

mkdir -p $experiment_dir
time python -m $module_name.utils.run $dataset_dir $experiment_dir/output.txt

# To produce visualization videos
# python -m $module_name.utils.visualize $dataset_dir $dataset_dir/list_video_id.txt $experiment_dir
