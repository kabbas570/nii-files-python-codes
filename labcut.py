from deeplabcut.modelzoo.video_inference import video_inference_superanimal
videos = ['/content/drive/MyDrive/Latest_DeepLabCut/videos/vid1/An1_2_30minutes_IL_31_5.mp4']
out_dir = '/content/drive/MyDrive/Latest_DeepLabCut/videos/out_dir1/'
superanimal_name = "superanimal_topviewmouse"
videotype = "mp4"
scale_list = [300]
video_inference_superanimal(
    videos,
    superanimal_name,
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    scale_list=[300],
    videotype="mp4",
    video_adapt=False,
    plot_trajectories=False,
    plot_bboxes=False,
    batch_size=8,               # adjust based on GPU memory
    detector_batch_size=8,      # adjust based on GPU memory
    device="cuda",              # force GPU usage
    max_individuals=1,
)
