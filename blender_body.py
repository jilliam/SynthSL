from copy import deepcopy
import io
import os
import random
import pickle
import sys
import array
import bpy, _cycles
from sacred import Experiment
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
from mathutils import Matrix

import torch

root = '.'
sys.path.insert(0, root)
mapping_path = os.environ.get('MAP_FILES_LOCATION', None)
smplerx_path = os.environ.get('SMPLERX_LOCATION', None)
smpl_models = os.environ.get('SMPL_MODELS_LOCATION', None)
assets_path = os.environ.get('ASSETS_LOCATION', None)
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

if mapping_path is None:
    raise ValueError('Environment variable MAP_FILES_LOCATION not defined'
                     'Please follow the instructions in Readme.md')

if smplerx_path is None:
    raise ValueError('Environment variable SMPLERX_LOCATION not defined'
                     'Please follow the instructions in Readme.md')

if smpl_models is None:
    raise ValueError('Environment variable SMPL_MODELS_LOCATION not defined'
                     'Please follow the instructions in Readme.md')

if assets_path is None:
    raise ValueError('Environment variable ASSETS_LOCATION not defined'
                     'Please follow the instructions in Readme.md')

from body_render import (mesh_manip, render, texturing, conditions,
                          imageutils, camutils, coordutils, depthutils,
                          getextures)

import smplx

ex = Experiment('generate_dataset')

color_map = [(161, 138, 122), (89, 49, 109), (211, 190, 222), (186, 130, 84), 
             (158, 80, 90), (137, 156, 208),  (168, 211, 243), (80, 60, 136), 
             (102, 63, 140), (34, 83, 190), (235, 188, 122), (96, 166, 217), 
             (110, 217, 230), (172, 117, 159), (110, 193, 167), (200, 200, 200), 
             (187, 247, 171), (106, 235, 143), (18, 118, 3), (238, 243, 77), 
             (42, 160, 62), (223, 156, 115), (141, 231, 150), (106, 158, 9), 
             (47, 113, 44), (229, 104, 90), (98, 144, 241), (102, 63, 140),
             (71, 226, 231), (132, 33, 35), (181, 35, 101), (244, 165, 208),
             (244, 57, 66), (168, 152, 167), (3, 195, 95), (96, 73, 58),
             (242, 250, 124), (167, 29, 5), (250, 233, 57), (20, 222, 114),
             (222, 34, 133), (199, 124, 31), (140, 106, 150), (9, 30, 52),
             (132, 133, 32), (112, 249, 39), (254, 231, 176), (99, 205, 230),
             (78, 245, 210), (149, 114, 185), (50, 146, 204), (172, 185, 54),
             (154, 166, 120), (203, 108, 19), (251, 134, 211), (1, 106, 67),
             (110, 241, 115), (112, 225, 118), (147, 47, 243), (63, 235, 92),
             (245, 214, 1), (166, 10, 135), (223, 139, 31), (27, 51, 179),
             (126, 37, 167), (26, 223, 29), (27, 65, 93), (147, 209, 254),
             (16, 148, 69), (123, 57, 29), (19, 143, 41), (211, 65, 90),
             (238, 220, 12), (161, 212, 98), (157, 237, 127), (62, 65, 145),
             (197, 128, 125), (182, 184, 44), (76, 210, 19), (75, 18, 137),
             (19, 36, 106), (146, 241, 54), (254, 203, 126), (123, 181, 237),
             (130, 138, 64), (54, 69, 186), (244, 223, 97), (86, 2, 23),
             (25, 61, 225), (167, 50, 24), (52, 247, 170), (41, 255, 149),
             (31, 177, 29), (131, 124, 21), (219, 209, 238), (162, 57, 174),
             (76, 115, 193), (157, 4, 226), (39, 89, 55), (241, 44, 49),
             (190, 78, 126), (204, 164, 246), (88, 54, 210), (128, 158, 157),
             (88, 248, 83), (144, 30, 249), (204, 87, 21), (215, 225, 220),
             (69, 206, 53), (201, 235, 20), (251, 65, 147), (165, 0, 193),
             (106, 132, 197), (0, 188, 255), (43, 129, 122), (86, 90, 126),
             (210, 78, 49), (249, 200, 233), (238, 169, 185), (158, 82, 12),
             (197, 50, 250), (29, 214, 192), (123, 148, 56), (209, 120, 178),
             (21, 22, 113), (52, 118, 119), (24, 94, 225), (246, 217, 177),
             (25, 56, 30), (178, 77, 156), (252, 6, 56), (228, 4, 63),
             (98, 99, 177), (93, 168, 91), (113, 235, 232), (125, 88, 72),
             (2, 209, 112), (13, 54, 30), (134, 194, 225), (229, 45, 17),
             (104, 235, 18), (65, 81, 44), (35, 165, 171), (66, 32, 80),
             (36, 101, 23), (104, 4, 49), (120, 60, 149), (139, 146, 19),
             (229, 80, 90), (24, 232, 208), (4, 129, 140), (63, 70, 206),
             (196, 165, 123), (134, 30, 119), (153, 247, 28), (130, 17, 57),
             (214, 146, 213), (9, 171, 226), (103, 152, 80), (254, 158, 107),
             (191, 236, 1), (110, 128, 5), (97, 241, 206), (51, 92, 187),
             (218, 167, 145), (36, 0, 17), (39, 251, 128), (14, 31, 254),
             (211, 35, 14), (253, 153, 82), (184, 124, 153), (219, 13, 105),
             (215, 127, 119), (229, 161, 138), (247, 136, 29), (188, 83, 159),
             (198, 173, 65), (98, 142, 32), (80, 95, 29), (161, 83, 150),
             (21, 55, 140), (101, 48, 39), (40, 171, 108), (188, 192, 206),
             (230, 243, 169), (171, 238, 39), (184, 207, 158), (123, 113, 126),
             (210, 170, 54), (129, 216, 87), (31, 32, 229), (121, 212, 135),
             (48, 58, 26), (158, 195, 159), (162, 8, 166), (252, 98, 196),
             (116, 229, 151), (220, 112, 120), (160, 119, 64), (110, 55, 125),
             (230, 164, 150), (145, 183, 116), (244, 6, 87), (240, 135, 124),
             (142, 187, 98), (148, 147, 131), (161, 69, 156), (168, 27, 246),
             (62, 107, 110), (126, 194, 162), (83, 126, 111), (16, 22, 106),
             (32, 51, 178), (174, 16, 151), (225, 107, 64), (217, 53, 122),
             (73, 166, 241), (18, 20, 93), (127, 184, 5), (15, 102, 170),
             (120, 58, 219), (82, 80, 132), (26, 99, 236), (5, 203, 237),
             (30, 199, 87), (44, 6, 126), (44, 7, 201), (206, 50, 18),
             (241, 124, 213), (158, 70, 105), (233, 111, 253), (209, 158, 150),
             (117, 75, 133), (215, 117, 68), (190, 32, 89), (147, 32, 50),
             (208, 71, 237), (230, 34, 43), (128, 63, 103), (65, 87, 27),
             (11, 243, 178), (105, 42, 1), (143, 139, 26), (106, 185, 161),
             (15, 54, 100), (23, 0, 188), (49, 101, 199), (116, 111, 8),
             (166, 37, 253), (112, 202, 115), (249, 175, 165), (205, 98, 51)]


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

@ex.config
def exp_config():
    bg_path = os.path.join(assets_path, 'backgrounds/blue.png')
    # Path to smplxer-x pkl output
    pkls_path = smplerx_path

    # folder containing pkl files or folder containing folders containing pkls files (latter one is the default smplify-x output)
    is_file_path = False
    # Half body (true) or full body (false)
    is_cropped = True

    # Height and width
    shape_y = 512
    shape_x = 512

    # Path to folder where to render
    results_root = 'results'

    # in ['train', 'test', 'val']
    split = 'train'

    # Idx of first frame
    frame_start = 0

    # Min and max distance to camera
    z_min = 2 # 0.5
    z_max = 3.5 # 0.8

    # Zoom to increase resolution of textures
    texture_zoom = 1

    # Combination of [imagenet|lsun|pngs|jpgs|with|4096]
    # texture_type = ['bodywithands']
    texture_type = ['meshcapade']

    # Texture path
    gender='female'
    texture_path = os.path.join(assets_path, 'textures')
    texture_clothing = os.path.join(texture_path, 'meshcapade/textures_clothing/smplx_textures_clothing_2048_20220905a')
    if gender == 'female':
        texture_body = '23_female_nongrey_female_0027.jpg'
    else:
        texture_body = '23_male_nongrey_male_0477.jpg'

    # Render full bodys and save body annotation
    render_body = True
    high_res_hands = False

    # Combination of [black|white|imagenet|lsun]
    background_datasets = ['imagenet', 'lsun']

    # Paths to background datasets
    lsun_path = '/sequoia/data2/gvarol/datasets/LSUN/data/img'
    imagenet_path = '/sequoia/data3/datasets/imagenet'

    # Lighting ambient mean and add
    ambient_mean = 0.9 # 0.7
    ambient_add = 0.1 # 0.5

    # body and hand params
    pca_comps = 12 # 45
    num_betas = 16  # 10
    # Pose params are uniform in [-hand_pose_var, hand_pose_var]
    hand_pose_var = 2

    # Path to fit folder
    smplx_model_path = os.path.join(smpl_models, 'smplx')


@ex.automain
def run(_config, results_root, split, frame_start, z_min, z_max,
        render_body, bg_path, texture_path, texture_body,
        texture_zoom, texture_type, texture_clothing, high_res_hands,
        background_datasets, lsun_path, imagenet_path, ambient_mean, ambient_add,
        smplx_model_path, num_betas, gender, hand_pose_var, pca_comps,
        is_cropped, is_file_path, pkls_path, shape_y, shape_x):

    if gender == 'female':
        body_model = 'SMPLX_FEMALE.npz'
    elif gender == 'male':
        body_model = 'SMPLX_MALE.npz'
    else:
        body_model = 'SMPLX_NEUTRAL.npz'

    smplx_model_path = os.path.join(smplx_model_path, body_model)

    # Set and create results folders
    seq_name = os.path.dirname(smplerx_path)
    name = os.path.basename(seq_name)
    results_root=os.path.join(results_root, name)
    folder_meta = os.path.join(results_root, 'meta')
    folder_rgb = os.path.join(results_root, 'rgb')
    folder_segm = os.path.join(results_root, 'segm')
    folder_temp_segm = os.path.join(results_root, 'tmp_segm')
    folder_depth = os.path.join(results_root, 'depth')
    folder_normal = os.path.join(results_root, 'tmp_normal')
    folder_gtflow = os.path.join(results_root, 'gtflow')
    folder_gtflow_img = os.path.join(results_root, 'gtflow_img')
    folders = [
        folder_meta, folder_rgb, folder_depth, folder_segm, folder_temp_segm,
        folder_normal, folder_gtflow, folder_gtflow_img
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    scene = bpy.data.scenes['Scene']
    bpy.ops.object.delete()         # Clear default scene cube

    # Attempt to set GPU device types if available
    bpy.context.scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    for compute_device_type in ('CUDA'):  #, 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            break
        except TypeError:
            pass

    # Enable all CPU and GPU devices
    cprefs.get_devices()
    # print("devices", cprefs.devices)
    for device in cprefs.devices:
        device.use = True
        # print(device, device["use"])

    # Set camera rendering params 210x260
    camutils.set_camera()
    scene.render.resolution_x = shape_x
    scene.render.resolution_y = shape_y
    scene.render.resolution_percentage = 100

    # Get full body textures
    body_textures = imageutils.get_bodytexture_paths(
        texture_type,
        split=split,
        body_textures_folder=os.path.join(texture_path, texture_type[0]),
        lsun_path=lsun_path,
        imagenet_path=imagenet_path,
        gender=gender)
    # print('Got {} body textures'.format(len(body_textures)))

    # Select random texture
    texture_body = os.path.basename(random.choice(body_textures))
    if texture_type[0] == 'meshcapade':
        eye_path = os.path.join(texture_path, texture_type[0], 'body_textures', 'eye', 'SMPLX_eye.png')
        texture_clothes = imageutils.get_clothing_texture_paths(textures_clothing=texture_clothing,
                                                                  gender=gender)
        text_clothes = random.choice(texture_clothes)
        texture_path = os.path.join(texture_path, texture_type[0], 'body_textures', 'smpl', 'MC_texture_skintones', gender)
    else:
        eye_path = os.path.join(texture_path, texture_type[0], 'eye.png')
        texture_path = os.path.join(texture_path, texture_type[0], split)

    # Get high resolution hand textures
    if high_res_hands:
        hand_textures = imageutils.get_hrhand_paths(texture_type, split=split)
        print('Got {} high resolution hand textures'.format(
            len(hand_textures)))
    print('Finished loading textures')

    # Load SMPL-X model
    smplx_model = smplx.create(smplx_model_path, model_type='smplx', num_betas=num_betas,
                               gender=gender, use_pca=False)
                               # gender=gender, use_pca=True, num_pca_comps=pca_comps)

    # Load and smooth the edges of the body model
    smplx_fbx = os.path.join(assets_path, 'models', 'SMPLX_{}.fbx'.format(gender))
    smplx_obj = mesh_manip.load_smplx(gender=gender, template=smplx_fbx)
    bpy.ops.object.shade_smooth()

    # Get camera info
    cam_calib = np.array(camutils.get_calib_matrix())
    cam_extr = np.array(camutils.get_extrinsic())
    # cam_extr1, R_world2cv1 = camutils.get_extrinsic()
    # R_world2cv = np.array(R_world2cv1)
    # cam_extr = np.array(cam_extr1)

    scs, materials, sh_path = texturing.initialize_texture(
        smplx_obj, path_assets=assets_path, texture_zoom=texture_zoom, tmp_suffix='tmp')

    # sh_coeffs = texturing.get_sh_coeffs(ambiant_mean=ambient_mean, ambiant_max_add=ambient_add)
    sh_coeffs = [ 1.1632192, -0.10256728, -0.17147624, 0.44304579, -0.10870888, 0.15684667, 0.3724881, -0.13730484, 0.52092402 ]
    texturing.set_sh_coeffs(scs, sh_coeffs)

    trans=None
    SMPLerX_files = []
    for file in os.listdir(smplerx_path):
        if file.endswith('.npz'):
            SMPLerX_files.append(file)
    SMPLerX_files.sort()

    print('Starting loop !')
    betas = torch.randn([1, smplx_model.num_betas], dtype=torch.float32)
    for i, eachone in enumerate(SMPLerX_files):

        # Load output from SMPLer-X
        file_name = os.path.join(smplerx_path, eachone)
        dataSMPLerX = np.load(file_name)

        # betas = dataSMPLerX["betas"]
        body_pose = dataSMPLerX["body_pose"]
        hand_pose_l = dataSMPLerX["left_hand_pose"]
        hand_pose_r = dataSMPLerX['right_hand_pose']
        global_orient = dataSMPLerX['global_orient']
        jaw_pose = dataSMPLerX['jaw_pose']
        expression = np.squeeze(dataSMPLerX['expression'], axis=0)
        camera_trans = dataSMPLerX['transl'][0]
        hand_pose_offset = 0

        smplx_verts, posed_model, meta_info, trans = mesh_manip.randomized_verts_smplx(
            smplx_model.cuda(), # smplx_model.cpu(),
            trans=trans,
            hand_pose_l=hand_pose_l,
            hand_pose_r=hand_pose_r,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            expression=expression,
            betas=betas,
            camera_trans=camera_trans,
            is_cropped=is_cropped,
            gender=gender,
            hand_pose_offset=hand_pose_offset)

        # Alter the fbx mesh **
        mesh_manip.alter_mesh(smplx_obj, smplx_verts.tolist())

        body_info = coordutils.get_full_body_info(
            posed_model,
            render_body=render_body,
            cam_extr=cam_extr,
            # R_world2cv=R_world2cv,
            cam_calib=cam_calib)
        body_infos = {**body_info, **meta_info}

        frame_idx = frame_start + i
        np.random.seed(frame_idx)
        random.seed(frame_idx)
        frame_prefix = '{:08}'.format(frame_idx)

        # camutils.set_camera()
        camera_name = 'Camera'
        depth_path = os.path.join(folder_depth, frame_prefix)
        normal_path = os.path.join(folder_normal, frame_prefix)
        gtflow_path = os.path.join(folder_gtflow, frame_prefix)
        tmp_segm_path = render.set_cycle_nodes(scene,
                                               bg_path,
                                               segm_path=folder_temp_segm,
                                               depth_path=depth_path,
                                               normal_path=normal_path,
                                               gtflow_path=gtflow_path)

        tmp_files = [tmp_segm_path]  # Keep track of temporary files to delete at the end
        tmp_depth = depth_path + '{:04d}.exr'.format(1)
        tmp_files.append(tmp_depth)
        tmp_normal = normal_path + '{:04d}.exr'.format(1)
        
        body_infos['texture_body'] = texture_body
        if texture_type[0] == 'meshcapade':
            body_infos['text_clothes'] = os.path.basename(text_clothes)
            tex_path = os.path.join(texture_path, 'skin', texture_body)
            tex_path = getextures.get_clothes_overlapped(tex_path, clothes_path=text_clothes)
        else:
            tex_path = os.path.join(texture_path, texture_body)  # body_textures[34]
        tex_path = getextures.get_eyes_overlapped(tex_path, eye_path=eye_path)
    
        # Replace high-res hands
        if high_res_hands:
            old_state = random.getstate()
            old_np_state = np.random.get_state()
            hand_path = random.choice(hand_textures)
            tex_path = texturing.get_overlaped(tex_path, hand_path)
            tmp_files.append(tex_path)
            # Restore previous seed state to not interfere with randomness
            random.setstate(old_state)
            np.random.set_state(old_np_state)

        # Update body+hands image
        tex_img = bpy.data.images.load(tex_path)
        for part, material in materials.items():
            material.node_tree.nodes['Image Texture'].image = tex_img

        # Render
        img_path = os.path.join(folder_rgb, '{}.jpg'.format(frame_prefix))
        scene.render.filepath = img_path
        scene.render.image_settings.file_format = 'JPEG'
        bpy.ops.render.render(write_still=True)

        camutils.check_camera(camera_name=camera_name)
        segm_img = cv2.imread(tmp_segm_path)  #[:, :, :]
        for j in range(0, scene.render.resolution_y - 1):
            for k in range(0, scene.render.resolution_x - 1):
                segm_img[j][k][0], segm_img[j][k][1], segm_img[j][k][
                    2] = color_map[segm_img[j][k][0]]

        if render_body:
            keep_render = True
        else:
            keep_render = conditions.segm_condition(segm_img, side=side, use_grasps=False)
        depth, depth_min, depth_max = depthutils.convert_depth(tmp_depth)

        body_infos['depth_min'] = depth_min
        body_infos['depth_max'] = depth_max
        body_infos['bg_path'] = bg_path
        body_infos['sh_coeffs'] = sh_coeffs
        # body_infos['body_tex'] = tex_path

        # Clean residual files
        if keep_render:
            # Write depth image
            final_depth_path = os.path.join(folder_depth, '{}.png'.format(frame_prefix))
            cv2.imwrite(final_depth_path, depth)

            # Save meta
            meta_pkl_path = os.path.join(folder_meta, '{}.pkl'.format(frame_prefix))
            with open(meta_pkl_path, 'wb') as meta_f:
                pickle.dump(body_infos, meta_f)

            # Write segmentation map
            segm_save_path = os.path.join(folder_segm, '{}.png'.format(frame_prefix))
            cv2.imwrite(segm_save_path, segm_img)
            ex.log_scalar('generated.idx', frame_idx)
        else:
            os.remove(img_path)
        for filepath in tmp_files:
            os.remove(filepath)
    print('DONE')


