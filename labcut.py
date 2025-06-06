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
