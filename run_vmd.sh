#!/bin/bash
filename=$1
cl=$2
cr=$3
fi=$(basename $filename)
file=${fi%.*}
time=$(tail -1 ../video_info/$file.txt)
python ../VMD_3D_Pose_Baseline_Multi-Objects/applications/pos2vmd_multi_modify_angle.py -v 2 -t1 ../$time/3d-pose-output -t2 ../$time/depth-output -t3 ../$time/vmd-output -b ../VMD_3D_Pose_Baseline_Multi-Objects/applications/animasa_miku_born.csv -c 30 -z 6 -x 0 -m 0 -i 0 -d 0 -a 1 -k 1 -e 0 -cl $cl -cr $cr
cp ../$time/vmd-output/output_video.vmd ./static/models/mmd/vmds
mv ./static/models/mmd/vmds/output_video.vmd ./static/models/mmd/vmds/$file"_"$cl"_"$cr.vmd
