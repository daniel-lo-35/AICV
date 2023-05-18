from help_scripts.python_scripts.COLMAP_functions import *
from help_scripts.python_scripts.estimate_plane import *
from help_scripts.python_scripts.color_virtual_image import *
from help_scripts.python_scripts.undistortion import compute_all_maps
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv

# import multiprocessing as mp
import argparse
import time
from tqdm import tqdm
from tqdm.contrib import tzip

import pupil_apriltags as apriltag

import pyransac3d as pyrsc

def main(args):
    # print('CPU count: ', mp.cpu_count())

    # Perform the reconstruction to get data
    #automatic_reconstructor()
    # image_undistorter()
    # stereo_fusion()

    # image_dir = r'/Users/ludvig/Documents/SSY226 Design project in MPSYS/Image-stitching-with-COLMAP/COLMAP_w_CUDA/'
    # image_dir = r'/home/r09521612/stitching/RCB2_S1_0416_01/AIDT-colmap/'
    # image_dir = args.colmap_dir
    # cameras, points3D, images = get_data_from_binary(image_dir)
    start = time.time()
    cameras, points3D, images = get_data_from_binary(args.colmap_dir)
    
    print(cameras)
    print(images)
    print('====================================')
    print('num_cameras: {}'.format(len(cameras)))
    print('num_images: {}'.format(len(images)))
    print('num_points3d: {}'.format(len(points3D)))
    print('====================================')

    coordinates = []
    for key in points3D:
        coordinates.append(points3D[key].xyz)
    coordinates = np.asarray(coordinates)
    end = time.time()
    print()
    print('################## Time for loading from COLMAP:', end-start, 'seconds ##################')
    print()

    start = time.time()
    # #Estimate a floor plane
    # plane, _ = ransac_find_plane(coordinates, threshold=0.01)
    # np.save(os.path.join(args.output_dir, 'plane.npy'), plane)
    # plane = None
    plane1 = pyrsc.Plane()
    plane, best_inliers = plane1.fit(coordinates, 0.01)
    end = time.time()
    print()
    print('################## Time for finding plane from RANSAC:', end-start, 'seconds ##################')
    print()

    # Get all camera matrices and images
    camera_intrinsics = {}
    all_camera_matrices = {}
    imgs = {}
    #image_dir = '../COLMAP_w_CUDA/images/'
    # image_dir = '../COLMAP_w_CUDA/dense/0/images/'

    xys = {}
    points3D_ids = {}

    # maps = compute_all_maps(image_dir, full_size_img=False)

    # real_image_dir = r'/home/r09521612/stitching/RCB2_S1_0416_01/Raw/'
    # real_image_dir = args.image_dir

    # Rearrange COLMAP data
    for key in tqdm(images.keys()):
        # print('cameraid, name', images[key].camera_id, cameras[key].id)
        # print('imageid, cameraid', key, images[key].camera_id)
        # imgs[images[key].camera_id] = np.asarray(plt.imread(image_dir+"images/" + images[key].name))
        # imgs[key] = np.asarray(plt.imread(real_image_dir + images[key].name))
        # imgs[key] = np.asarray(plt.imread(args.image_dir + images[key].name))
        imgs[key] = cv.imread(os.path.join(args.image_dir, images[key].name))

        # map_x, map_y = maps[key]
        # imgs[images[key].camera_id] = cv.remap(imgs[images[key].camera_id], map_x, map_y, cv.INTER_LANCZOS4)
        # imgs[key] = cv.remap(imgs[key], map_x, map_y, cv.INTER_LANCZOS4)

        # all_camera_matrices[images[key].camera_id] = camera_quat_to_P(images[key].qvec, images[key].tvec)
        all_camera_matrices[key] = camera_quat_to_P(images[key].qvec, images[key].tvec)
        # camera_intrinsics[cameras[key].id] = cameras[key]
        # camera_intrinsics[images[key].camera_id] = cameras[images[key].camera_id]
        camera_intrinsics[key] = cameras[images[key].camera_id]

        xys[key] = images[key].xys
        points3D_ids[key] = images[key].point3D_ids

    # VIRTUAL CAMERA WITH MEAN CENTER OF OTHER CAMERAS AND PRINCIPAL AXIS AS PLANE NORMAL
    start = time.time()
    Pv = create_virtual_camera(all_camera_matrices,plane)
    # w = 500
    # h = 500
    # f = 250
    # w = 4000
    # h = 4000
    w = args.dimension
    h = args.dimension
    # # f = 1000
    # f = 0.16 * args.dimension
    # f = 0.14 * args.dimension
    f = args.zoom_size * args.dimension
    # K_virt = np.asarray([[f, 0, w/2],[0, f, h/2],[0, 0, 1]])
    # f, cx, cy, _ = cameras[1].params
    # K_virt = np.asarray([[f, 0, cx],[0, f, cy],[0, 0, 1]])
    K_virt = np.asarray([[f, 0, w/2],[0, f, h/2],[0, 0, 1]])

    print('##### Virtual Camera #####')
    print('Pv:', Pv)
    print('K_virt:', K_virt)
    print()
    end = time.time()
    print()
    print('################## Time for creating virtual camera:', end-start, 'seconds ##################')
    print()


    ################## Below code is to get rgb from points3D and project onto the virtual camera. ##################
    ################## Code modified from: https://github.com/XYZ-qiyh/colmap_sparse_recon/blob/main/colmap_sparse_to_depth.py ##################
    start = time.time()

    # # 1. remove invisible points
    # xys_v = image.xys[image.point3D_ids > -1]
    # point3D_ids_v = image.point3D_ids[image.point3D_ids > -1]

    # 2. get corresoponding 3D points
    XYZ_world = []
    RGB_world = []
    points3D_keys = []
    # for idx in point3D_ids_v:
    for idx in points3D:
        XYZ_world.append(points3D[idx].xyz)
        RGB_world.append(points3D[idx].rgb)
        points3D_keys.append(idx)

    XYZ_world = np.array(XYZ_world)
    RGB_world = np.array(RGB_world)

    # print("XYZ_world.shape:", XYZ_world.shape)
    # print("RGB_world.shape:", RGB_world.shape)
    # print("XYZ_world[0]:", XYZ_world[0])
    # print("RGB_world[0]:", RGB_world[0])

    # 3. [R|t] transform XYZ_world to XYZ_cam
    # R = qvec2rotmat(image.qvec)
    # t = image.tvec
    R, t = np.hsplit(Pv, [3])
    t = t.flatten()
    print("R:", R)
    print("t:", t)

    XYZ_cam = np.matmul(R, XYZ_world.transpose()) + t[:, np.newaxis]
    XYZ_cam = XYZ_cam.transpose()

    # 4. get the depth value
    depth_values = XYZ_cam[:, 2] # 3rd component

    print("XYZ_cam.shape:", XYZ_cam.shape)
    print("XYZ_cam[0]:", XYZ_cam[0])

    # 5.  project the 3d points to 2d pixel coordinate
    x_norm = XYZ_cam[:, 0] / XYZ_cam[:, 2]
    y_norm = XYZ_cam[:, 1] / XYZ_cam[:, 2]

    new_w = args.dimension
    new_h = args.dimension
    new_fx = f
    new_fy = f
    new_cx = w/2
    new_cy = h/2

    x_2d = x_norm * new_fx + new_cx
    y_2d = y_norm * new_fy + new_cy

    # save sparse depth map
    if args.background == "black":
        depth_map = np.zeros((new_h, new_w, 3), dtype=np.float32)
    elif args.background == "white":
        depth_map = np.ones((new_h, new_w, 3), dtype=np.float32)*255
    else:
        print("args.background ({}) not recognized! Default to white background".format(args.background))
        depth_map = np.ones((new_h, new_w, 3), dtype=np.float32)*255

    x_2d_float = np.copy(x_2d)
    y_2d_float = np.copy(y_2d)
    x_2d = np.round(x_2d).astype(np.int32)
    y_2d = np.round(y_2d).astype(np.int32)

    # for x, y, z in zip(x_2d, y_2d, depth_values):
    for x, y, z in tzip(x_2d, y_2d, RGB_world):
        if (x < 0) or (y < 0) or (x >= new_w) or (y >= new_h):
            continue
        depth_map[(y, x)] = z
        # print("depth: {}".format(z))
    # print("depth_map: {}".format(depth_map))

    # report density
    if args.background == "black":
        print("pct: {:.2f}%".format(100*(depth_map>0).mean()))
    else:
        print("pct: {:.2f}%".format(100*(depth_map<255).mean()))

    cv.imwrite(os.path.join(args.output_dir, 'detailed_results', '{}-sparse_to_rgb.jpg'.format(args.output_name)), depth_map)

    XY_virt = np.column_stack((x_2d, y_2d))
    XY_virt_float = np.column_stack((x_2d_float, y_2d_float))

    end = time.time()
    print()
    print('################## Time for converting sparse to rgb:', end-start, 'seconds ##################')
    print()


    # TEST WITH EXISTING CAMERA
    # K_temp, dist_temp = build_intrinsic_matrix(camera_intrinsics[1])
    # Pv = all_camera_matrices[1]['P']
    # K_virt = K_temp
    # w = int(K_virt[0, 2]*2)
    # h = int(K_virt[1, 2]*2)

    # TEST HOMOGRAPHY 2.0
    H = {}
    P_real_new = {}
    score = {}
    K = {}

    # K_temp, dist_temp = build_intrinsic_matrix(camera_intrinsics[1])
    start = time.time()
    for key in tqdm(all_camera_matrices):
        # print('Key vs cam id', key, camera_intrinsics[key].id)
        K[key], dist_temp = build_intrinsic_matrix(camera_intrinsics[key])
        H[key],plane_new,P_real_new[key],P_virt_trans = compute_homography(Pv, all_camera_matrices[key]['P'], K_virt, K[key], plane)
        
        # What's NEW: Create a score to sort cameras by distance w/ virtual camera
        score[key] = abs(np.prod(all_camera_matrices[key]['P'] - Pv))
    
    score = dict(sorted(score.items(), key = lambda x:x[1]))
    end = time.time()
    print()
    print('################## Time for finding homography:', end-start, 'seconds ##################')
    print()

    print('HOMO',H)# color image
    # # color_images, stitched_image = color_virtual_image(plane, Pv, w, h, imgs, all_camera_matrices, camera_intrinsics, K_virt,'homography',H)
    # stitched_image = color_virtual_image(plane, Pv, w, h, imgs, all_camera_matrices, camera_intrinsics, K_virt, args.method, H, score, args.sort_by, args.skip, args.quality)
    # print('stitched_image:', stitched_image)
    # print('shape of stitched image:', stitched_image.shape)
    # stitched_image = stitched_image/255
    # imgplot = plt.imshow(stitched_image)
    # plt.imsave(os.path.join(args.output_dir, 'stitched_image.jpg'), stitched_image)
    # # plt3d = plot_3D(points3D,plane,all_camera_matrices,Pv)

    imgs_transform = []
    imgs_transform_crop = []
    corner_transform_list_0 = []
    corner_transform_list_1 = []
    corner_transform_list_2 = []
    corner_transform_list_3 = []

    start = time.time()
    for index in tqdm(imgs):
        matrix = np.linalg.inv(K[index])
        matrix = np.matmul(np.linalg.inv(H[index]), matrix)
        matrix = np.matmul(K_virt, matrix)
        matrix = matrix / matrix[-1][-1]

        shape = imgs[index].shape
        # crop_size = 1000
        crop_size = shape[0] // 2
        # crop_size = shape[0]
        im_out = cv.warpPerspective(imgs[index], matrix, (args.dimension, args.dimension))
        if args.verbose:
            cv.imwrite(os.path.join(args.output_dir, 'detailed_results', '{}-{:d}-transform.jpg'.format(args.output_name, index)), im_out)
        imgs_transform.append(im_out)
        
        if args.crop:
            im_out_crop = cv.warpPerspective(np.concatenate((np.zeros((shape[0]-crop_size, shape[1], shape[2]), dtype=np.uint8), imgs[index][-crop_size:])), matrix, (args.dimension, args.dimension))
            if args.verbose:
                cv.imwrite(os.path.join(args.output_dir, 'detailed_results', 'crop-{}-{:d}-transform.jpg'.format(args.output_name, index)), im_out_crop)
            imgs_transform_crop.append(im_out_crop)

        w_downsample = 2400
        april_tag = find_apriltag(imgs[index], args.apriltag_size, w_downsample)
        if april_tag is not None:
            for i, corner in enumerate(april_tag.corners):
                corner_transform = np.matmul(matrix, np.array((corner[0]/w_downsample*shape[1], corner[1]/w_downsample*shape[1], 1)))
                corner_transform = corner_transform / corner_transform[-1]
                if i == 0:
                    corner_transform_list_0.append(corner_transform)
                elif i == 1:
                    corner_transform_list_1.append(corner_transform)
                elif i == 2:
                    corner_transform_list_2.append(corner_transform)
                elif i == 3:
                    corner_transform_list_3.append(corner_transform)

    if len(corner_transform_list_0) > 0:
        c0 = (np.mean(corner_transform_list_0, axis = 0)).astype(int)[: : -1]
        c1 = (np.mean(corner_transform_list_1, axis = 0)).astype(int)[: : -1]
        c2 = (np.mean(corner_transform_list_2, axis = 0)).astype(int)[: : -1]
        c3 = (np.mean(corner_transform_list_3, axis = 0)).astype(int)[: : -1]

        pixel_distance = 0.25*(np.sum((c0-c1)**2)**0.5+np.sum((c1-c2)**2)**0.5+np.sum((c2-c3)**2)**0.5+np.sum((c3-c0)**2)**0.5)
        lengh_per_pixel = args.apriltag_size / pixel_distance
        print('Length per pixel:', lengh_per_pixel)
    else:
        lengh_per_pixel = None

    end = time.time()
    print()
    print('################## Time for trasnforming images using cv.warpPerspective + finding AprilTag:', end-start, 'seconds ##################')
    print()

    # imgs_transform.reverse()

    start = time.time()
    output_image = np.zeros(imgs_transform[0].shape)

    # New method: Use locations of points3D as reference for marking pieces of area to stitch.
    for i, key in enumerate(tqdm(images.keys())):
        # xys_image = xys[key]
        # empty_kp = []
        # points3D_ids_image = points3D_ids[key]
        xys_plane = []
        for i, point3d in enumerate(points3D_ids[key]):
            if point3d >= 0:
                xys_plane.append(XY_virt_float[points3D_keys.index(point3d)])
            # else:
            #     empty_kp.append(i)
        xys_plane = np.asarray(xys_plane)
        # xys_image = np.delete(xys_image, empty_kp, 0)
        im_kp = imgs_transform[i] # TODO: mask the transformed image into image pieces using the location of keypoints (xys_plane)
        if args.verbose:
            cv.imwrite(os.path.join(args.output_dir, 'detailed_results', '{}-{:d}-keypoints.jpg'.format(args.output_name, key)), im_kp)

    for image in tqdm(imgs_transform):    # Current sort image method: by input sequence. TODO: 1. Make it input according to parameters (distance? angle? matching points (amount & location)?) 2. Image pieces instead of whole image
        output_image = np.where(image, image, output_image)
    if args.crop:
        output_image_crop = np.zeros(imgs_transform_crop[0].shape)
        for image in tqdm(imgs_transform_crop):
            output_image_crop = np.where(image, image, output_image_crop)

    if args.background == "white":
        bg = np.ones(imgs_transform[0].shape)*255
        output_image = np.where(output_image, output_image, bg)
        if args.crop:
            output_image_crop = np.where(output_image_crop, output_image_crop, bg)

    cv.imwrite(os.path.join(args.output_dir, '{}.jpg'.format(args.output_name)), output_image)
    np.save(os.path.join(args.output_dir, '{}.jpg-length_per_pixel.npy'.format(args.output_name)), lengh_per_pixel)
    if args.crop:
        cv.imwrite(os.path.join(args.output_dir, 'crop-{}.jpg'.format(args.output_name)), output_image_crop)
        np.save(os.path.join(args.output_dir, 'crop-{}.jpg-length_per_pixel.npy'.format(args.output_name)), lengh_per_pixel)

    end = time.time()
    print('################## Time for making stitched image:', end-start, 'seconds ##################')



def find_apriltag(one_image, apriltag_size = 0.18, w_downsample = 2400):    # apriltag_size in meters
    # w_downsample = 2400
    resize_tuple = (w_downsample, int(one_image.shape[0] * w_downsample / one_image.shape[1]))
    # print(resize_tuple)
    img_resize = cv.resize(one_image, resize_tuple, interpolation=cv.INTER_LINEAR_EXACT)
    # img_resize = one_image
    gray_image = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
    tag_detector = apriltag.Detector(families='tag36h11', # tag25h9
                                    #  border=1,
                                        nthreads=2,
                                        quad_decimate=1.0,
                                    #  quad_blur=3,
                                    #  refine_edges=True,
                                    #  refine_decode=True,
                                    #  refine_pose=False,
                                    #  debug=False,
                                    #  quad_contours=True
                                    )
        
    # Perform apriltag detection to get a list of detected apriltag
    tags = tag_detector.detect(gray_image, estimate_tag_pose=False)
    # cam_params = [2.46634964e+03, 2.59200000e+03, 1.94400000e+03, 5.46740775e-04]
    # cam_params = [1000, 2000, 2000, 5.46740775e-04]
    # tags = tag_detector.detect(gray_image, estimate_tag_pose=True, camera_params = cam_params, tag_size=apriltag_size) 
    if len(tags) == 0: 
        print("Skipping since no AprilTag detected.")
        return None

    for i, tag in enumerate(tags):
        print()
        print('############ TAG', i, '############')
        print(tag)

    # one_tag = tags[0]
    # c0 = (one_tag.corners[0]).astype(int)[: : -1] / w_downsample * one_image.shape[1]
    # c1 = (one_tag.corners[1]).astype(int)[: : -1] / w_downsample * one_image.shape[1] 
    # c2 = (one_tag.corners[2]).astype(int)[: : -1] / w_downsample * one_image.shape[1]
    # c3 = (one_tag.corners[3]).astype(int)[: : -1] / w_downsample * one_image.shape[1]

    # pixel_distance = 0.25*(np.sum((c0-c1)**2)**0.5+np.sum((c1-c2)**2)**0.5+np.sum((c2-c3)**2)**0.5+np.sum((c3-c0)**2)**0.5)
    # lengh_per_pixel = apriltag_size/pixel_distance
    # print('Length per pixel:', lengh_per_pixel)
    # return one_tag, lengh_per_pixel
    return tags[0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--colmap_dir", help="colmap dir. Folder structure: <colmap_dir>/sparse/0/cameras.bin & points3D.bin & images.bin")
    parser.add_argument("-i", "--image_dir", help="image dir. Folder structure: <image_dir>/<IMG_NAME>.jpg")
    parser.add_argument("-o", "--output_dir", help="where to save the stitching results", default="output")
    parser.add_argument("-d", "--dimension", help="Size of output image", type=int, default=4000)
    parser.add_argument("-s", "--skip", help="Whether to skip images when building stitched images (may save time but sacrifice quality)", action="store_true", default=False)
    parser.add_argument("-q", "--quality", help="Produce higher quality images by only choosing pixels close to camera", action="store_true", default=False)
    parser.add_argument("-m", "--method", help="Choose method for stitching: 'homography' or 'ray_tracing'", default="homography")
    parser.add_argument("--sort_by", help="Choose method for sorting images during stitching: 'score' (closeness to virtual camera), 'default' (sequence of taking images), or 'inverse' (inverse sequence of taking images (WARNING: May have ERROR))", default="score")
    parser.add_argument("--apriltag_size", help="Length of one side of apriltag (in meters)", type=float, default=0.18)
    parser.add_argument("-v", "--verbose", help="Whether to save the transformed images", action="store_true", default=False)
    parser.add_argument("-b", "--background", help="Background color of result image", default="white")
    parser.add_argument("-z", "--zoom_size", help="Parameter to control the zooming of virtual camera. The smaller the parameter, the more wide the camera len is", type=float, default=0.16)
    parser.add_argument("--crop", help="Whether to save the cropped images", action="store_true", default=False)
    parser.add_argument("-n", "--output_name", help="File name of the stitching results", default="stitched_image")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "detailed_results"), exist_ok=True)
    main(args)
