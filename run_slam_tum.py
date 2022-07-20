import torch
import sys
import glob
import os
import csv
import cv2
import threading
from model import Camera, Mapper

def read_files(folder_path=r"/home/itsuki/RGBD/rgbd_dataset_freiburg1_teddy/"):
    csv_file = open(folder_path + "rgb.txt", "r")
    f = csv.reader(csv_file, delimiter=" ")
    next(f)
    next(f)
    next(f)
    rgb_filenames = []
    for row in f:
        rgb_filenames.append("{}{}".format(folder_path, row[1]))
    csv_file = open(folder_path + "depth.txt", "r")
    f = csv.reader(csv_file, delimiter=" ")
    next(f)
    next(f)
    next(f)
    depth_filenames = []
    for row in f:
        depth_filenames.append("{}{}".format(folder_path, row[1]))
    return rgb_filenames, depth_filenames

def mappingThread(mapper):
    while True:
        mapper.mapping()
        time.sleep(0.1)
        print("update map")
def main():
    mapper = Mapper()
    if 2<= len(sys.argv):
        rgb_filenames, depth_filenames = read_files(sys.argv[1]) #get filenames of rgbd captures. 
    else:
        rgb_filenames, depth_filenames = read_files()
    frame_length = min(len(rgb_filenames), len(depth_filenames)) # get the shorter of the 2 files
    mapper.addCamera(rgb_filenames[0], # use first frame to create a first camera
                    depth_filenames[0],
                    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0) #extrinsics set to 0 (origin frame)
    fixed_camera = Camera(cv2.imread(rgb_filenames[0], cv2.IMREAD_COLOR), 
                             cv2.imread(depth_filenames[0], cv2.IMREAD_ANYDEPTH), 
                    0.0,0.0,0.0,1e-8,1e-8,1e-8,0.0,0.0) #small nonzero rotation initialized
    tracking_camera = Camera(cv2.imread(rgb_filenames[0], cv2.IMREAD_COLOR), 
                             cv2.imread(depth_filenames[0], cv2.IMREAD_ANYDEPTH), 
                    0.0,0.0,0.0,1e-8,1e-8,1e-8,0.0,0.0)
    # Initialize Map
    for i in range(200): # loop 200 times over first image to really memorize it.
        mapper.mapping(batch_size=200, activeSampling=False) # sample 200 ray points in each pixel sample.

    # For calc kinematics for camera motion
    last_pose = tracking_camera.params #initial camera pose (we set to origin)
    camera_vel = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0]).detach().cuda().requires_grad_(True) #current cam velocity
    
    last_kf=0

    mapping_thread = threading.Thread(target=mappingThread, args=(mapper,))# create new threads that call mappingThread(mapper)
    # we run mapper update every 0.1s. In between, we can add cameras, so mapper than learn new views and re-estimate previous camera poses
    # along the way.
    mapping_thread.start()

    # now we can proceed to subsequent frames
    """ Starting from origin pose, each frame comes with an rgbd image and a velocity vector.
    We add the velocity vector to the 
    """
    for frame in range(1,frame_length):
        tracking_camera.params.data += camera_vel #update camera extrinsics with velocity (change in pose)
        tracking_camera.setImages(cv2.imread(rgb_filenames[frame], cv2.IMREAD_COLOR), # load next frame images
                                  cv2.imread(depth_filenames[frame], cv2.IMREAD_ANYDEPTH))
        pe = mapper.track(tracking_camera) # pass tracking camera with new pose and new image to mapper.track()
        # mapper.track loops and learns new pose and rgbd image to optimize lg & lp.
        camera_vel = 0.2 * camera_vel + 0.8*(tracking_camera.params-last_pose) # since we have updated our pose,
        # we now adjust camera_vel by retaining 20% of old velocity and 80% of updated velocity.
        # the update reduces tracking error as it takes into account map alignment.
        last_pose = tracking_camera.params # save last pose from current camera
        if pe < 0.65 and frame-last_kf>5: #if pe (recon_acc) is low and we are more than 5 frames ahead of last keyframe
            p = tracking_camera.params
            mapper.addCamera(rgb_filenames[frame], # we save the new frame as a keyframe in mapper
                    depth_filenames[frame],
                    last_pose[3],last_pose[4],last_pose[5],
                    last_pose[0],last_pose[1],last_pose[2],
                    last_pose[6],last_pose[7])
            print("Add keyframe")
            print(last_pose.cpu())
            last_kf=frame
        # Render from tracking camera
        mapper.render_small(tracking_camera, "view") # render and save image for this frame.
        # Render from fixed camera
        #mapper.render_small(fixed_camera, "fixed_camera")
    mapping_thread.join()

main()