#!/bin/bash
filename=$1
echo "$filename"
echo "build a directory"
time=$(date +%F_%H-%M-%S)
mkdir ../$time
mkdir ../$time/openpose-output
mkdir ../$time/3d-pose-output
mkdir ../$time/depth-output
mkdir ../$time/vmd-output
echo " start to predict 2d joint position"
python ../tf-pose-estimation-ver2/run_video_save4.py --model=cmu --video=${filename} --output_json=../${time}/openpose-output/json --output_video=../${time}/openpose-output/openpose.avi
echo "2d joint position finished"
rm -r ../3d-pose-baseline-vmd/openpose_output/*
cp -r ../$time/openpose-output/json/* ../3d-pose-baseline-vmd/openpose_output
cd ../3d-pose-baseline-vmd
echo "start to predict 3d joint position"
python src/openpose_3dpose_sandbox_vmd.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 200 --sample --load 4874200 --gif_fps 60
echo "3d joint position finished"
cd ..
cp -r ./3d-pose-baseline-vmd/openpose_output/movie_smoothing.gif ./$time/3d-pose-output
cp -r ./3d-pose-baseline-vmd/openpose_output/pos.txt ./$time/3d-pose-output
cp -r ./3d-pose-baseline-vmd/openpose_output/smoothed.txt ./$time/3d-pose-output
echo "start to predict depth information"
cd web_test
python ../deeper-depth-prediction/FCRN-DepthPrediction/tensorflow/predict_video_modify_path.py --model=../deeper-depth-prediction/FCRN-DepthPrediction/tensorflow/models/NYU_FCRN.ckpt --video_path=${filename} --baseline_path=../$time/3d-pose-output/smoothed.txt --interval=1 --verbose=2 --depth_path=../$time/depth-output
echo "depth prediction finished"
echo "start to generate vmd file"
cl=$2
cr=$3
python ../VMD_3D_Pose_Baseline_Multi-Objects/applications/pos2vmd_multi_modify_angle.py -v 2 -t1 ../$time/3d-pose-output -t2 ../$time/depth-output -t3 ../$time/vmd-output -b ../VMD_3D_Pose_Baseline_Multi-Objects/applications/animasa_miku_born.csv -c 30 -z 6 -x 0 -m 0 -i 0 -d 0 -a 1 -k 1 -e 0 -cl $cl -cr $cr
cp ../$time/vmd-output/output_video.vmd ./static/models/mmd/vmds
fi=$(basename $filename)
file=${fi%.*}
mv ./static/models/mmd/vmds/output_video.vmd ./static/models/mmd/vmds/$file"_"$cl"_"$cr.vmd
echo $time>>./../video_info/$file.txt
