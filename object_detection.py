def object_detection(video_path:str):
    
    """
    Python program to perform object detection on commercial videos using a Panoptic Segmentation model from detectron2  
    library.
    """

    # ============================================================
    # Import
    # ============================================================

    # Import internal modules
    import os

    # Import 3rd party modules
    import pandas as pd
    import numpy as np
    import cv2


    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog

    # ============================================================
    # Main functions
    # ============================================================

    df = pd.DataFrame(columns=["timestamp","label", "score", "top_left_x", "top_left_y", "bottom_right_x",
    "bottom_right_y"])
    print(df)

    # Extract video properties
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    print('frames per second: ', frames_per_second)
    print('{} frames per second = 1 frame every 33 ms'.format(frames_per_second))

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(num_frames)
    video_len = num_frames / frames_per_second
    print('video length: ', video_len, 'seconds')
    timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))


    # Initialize predictor
    cfg = get_cfg()
    # get config file from remote
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80  # set threshold for this model
    cfg.MODEL.WEIGHTS = './model_final_c10459.pkl' # load model
    predictor = DefaultPredictor(cfg)


    while True:
          hasFrame, frame = video.read()
          if not hasFrame:
              break

          # Get prediction results for this frame
          outputs = predictor(frame)
          panoptic_seg, segments_info = predictor(frame)["panoptic_seg"]


          length = len(outputs["instances"].pred_boxes) 
          if length == 0:
            timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))
            top_left_x = np.nan
            top_left_y = np.nan
            bottom_right_x = np.nan
            bottom_right_y = np.nan
            score = np.nan
            label = np.nan
            new_row = {

                "timestamp": timestamp, "label": label, "score": score,
                "top_left_x": top_left_x, "top_left_y": top_left_y, "bottom_right_x": bottom_right_x,
                "bottom_right_y": bottom_right_y}

            df = df.append(new_row, ignore_index=True, sort=None)
            print(df)
            print('-----------------')

          else:
            for i in range(len(outputs["instances"].pred_boxes)):

              # get bounding boxes
              for data in outputs["instances"].pred_boxes[i]:
                for coor in range(len(data)):
                  top_left_x = data[0].item()
                  top_left_y = data[1].item()
                  bottom_right_x = data[2].item()
                  bottom_right_y = data[3].item()
                timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))
                score = round(outputs["instances"].scores[i].item(), 2)
                num = outputs['instances'].pred_classes[i].item()
                label = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num]


                new_row = {

                "timestamp": timestamp, "label": label, "score": score,
                "top_left_x": top_left_x, "top_left_y": top_left_y, "bottom_right_x": bottom_right_x,
                "bottom_right_y": bottom_right_y}

                df = df.append(new_row, ignore_index=True, sort=None)
                print(df)
                print('-----------------')

            # get stuff labels
            for segment in segments_info:
              if segment.get('isthing') == False:
                timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))
                cat_id = segment.get('category_id')
                label = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[cat_id]
                print(label)
                new_row = {

                "timestamp": timestamp, "label": label, "score": np.nan,
                "top_left_x": np.nan, "top_left_y": np.nan, "bottom_right_x": np.nan,
                "bottom_right_y": np.nan}

                df = df.append(new_row, ignore_index=True, sort=None)
                print(df)
                print('-----------------')



    # Release resources
    video.release()
    df.to_csv('test.csv', index=False) # save dataframe to a csv file