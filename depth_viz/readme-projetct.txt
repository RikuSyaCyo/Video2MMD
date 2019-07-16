
1. openpose
当前路径：/userhome/2072/msp18027/tf-pose-estimation-ver2

run:python run_video_save4.py --model=cmu --video=./videos/yxj_com.mp4 --output_json=./result/result_json_yxj --output_video=./result/result_video/test_yxj.avi

input:./videos/
output:
       video:/userhome/2072/msp18027/tf-pose-estimation-ver2/result/result_video
       json:/userhome/2072/msp18027/tf-pose-estimation-ver2/result/result_json_xx


2.3d-coordinate-prediction
当前路径：/userhome/2072/msp18027/3d-pose-baseline-vmd

run:python src/openpose_3dpose_sandbox_vmd.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 200 --sample --load 4874200 --gif_fps 60

input:/userhome/2072/msp18027/3d-pose-baseline-vmd/openpose_output/下的json和txt文件
output：pos.txt:输出的3d坐标
        movie_smoothing.gif：3d坐标动图
        smoothed.txt：smoothed过的坐标


3.deeper-depth-prediction
当前路径： /userhome/2072/msp18027/deeper-depth-prediction/FCRN-DepthPrediction/tensorflow

run:python video_predict.py —model_path=‘./models/NYU_FCRN.ckpt’ —video_path=‘1.mp4’ —baseline=./ —interval=1 —verbose=2

预测image的深度：predict.py（predict_original.py源代码）
预测video的深度：video_predict.py（predict_video_original.py源代码）

模型位置：./models

input:smoothed.txt

output: moive_depth.gif
        ./depth: 每一帧的输出图片
        depth.txt

4. 3d-global-position
当前路径：3d-global-position

run: python 3d_global_position1.py

Input:pos.txt depth.txt

Output:movie_smoothing.gif
       result.png: 每一帧的3d全局坐标
       global_pos.txt 全局坐标输出

        



