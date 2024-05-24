from copy import deepcopy
import random

import numpy as np
import torch


def alter_mesh(obj, verts):
    import bmesh
    import bpy
    from mathutils import Vector
    # bpy.context.scene.objects.active = obj
    bpy.context.view_layer.objects.active = obj
    mesh = bpy.context.object.data

    bm = bmesh.new()

    # convert the current mesh to a bmesh (must be in edit mode)
    bpy.ops.object.mode_set(mode='EDIT')
    bm.from_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')  # return to object mode

    print("BM VERTS SHAPE",len(bm.verts))
    for v, bv in zip(verts, bm.verts):
        bv.co = Vector(v)

    # make the bmesh the object's mesh
    bm.to_mesh(mesh)
    bm.select_flush(True)
    bm.free()  # always do this when finished


def load_body_data(smpl_data, gender='female', idx=0, n_sh_bshapes=10):
    """
    Loads MoSHed pose data from CMU Mocap (only the given idx is loaded), and loads all CAESAR shape data.
    Args:
        smpl_data: Files with *trans, *shape, *pose parameters
        gender: female | male. CAESAR data has 2K male, 2K female shapes
        idx: index of the mocap sequence
        n_sh_bshapes: number of shape blendshapes (number of PCA components)
    """
    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))

    name = sorted(cmu_keys)[idx % len(cmu_keys)]

    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {
                'poses': smpl_data[seq],
                'trans': smpl_data[seq.replace('pose_', 'trans_')]
            }

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return (cmu_parms, fshapes, name)


def load_smpl(template='assets/models/basicModel_{}_lbs_10_207_0_v1.0.2.fbx',
              gender='f'):
    """
    Loads smpl model, deleted armature and renames mesh to 'Body'
    """
    import bpy
    filepath = template.format(gender)
    bpy.ops.import_scene.fbx(
        filepath=filepath, axis_forward='Y', axis_up='Z', global_scale=100)
    obname = '{}_avg'.format(gender)
    ob = bpy.data.objects[obname]
    ob.parent = None

    # print("Objects")
    # for elem in bpy.data.objects:
    #     print(elem.name)
    # print("Meshes")
    # for elem in bpy.data.meshes:
    #     print(elem.name)

    # Delete armature
    bpy.ops.object.select_all(action='DESELECT')
    # bpy.data.objects['Armature'].select = True
    bpy.data.objects['Armature'].select_set(True)
    bpy.ops.object.delete(use_global=False)

    # Rename mesh
    # bpy.data.meshes['Untitled'].name = 'Body'
    bpy.data.meshes['Mesh'].name = 'Body'
    return ob


# def load_smplx(template='assets/models/SMPLX_{}.fbx', gender='female'):
def load_smplx(gender='female', template='assets/models/SMPLX_{}.fbx'):
    """
    Loads smpl model, deleted armature and renames mesh to 'Body'
    """
    import bpy
    # print("Init")
    # for elem in bpy.data.meshes:
    #     print(elem.name)
    filepath = template.format(gender)
    print("Loading SMPL-X fbx:", filepath)
    bpy.ops.import_scene.fbx(
        filepath=filepath, axis_forward='Y', axis_up='Z', global_scale=100)

    # obname = '000000'
    obname = 'SMPLX-mesh-{}'.format(gender)
    ob = bpy.data.objects[obname]
    ob.parent = None

    # print("Objects")
    # for elem in bpy.data.objects:
    #     print(elem.name)
    # print("Meshes")
    # for elem in bpy.data.meshes:
    #     print(elem.name)

    # Delete armature
    bpy.ops.object.select_all(action='DESELECT')
    arm_name = 'SMPLX-{}'.format(gender)
    bpy.data.objects[arm_name].select_set(True)
    bpy.ops.object.delete(use_global=False)

    # print("Final")
    # for elem in bpy.data.meshes:
    #     print(elem.name)

    # Rename mesh
    mesh_name = 'SMPLX-shapes-{}.001'.format(gender)
    # mesh_name = 'SMPLX-shapes-{}'.format(gender)
    bpy.data.meshes[mesh_name].name = 'Body'

    return ob


def random_global_rotation():
    """
    Creates global random rotation in axis-angle rotation format.
    """
    # 1. We will pick random axis: random azimuth and random elevation in spherical coordinates.
    # 2. We will pick random angle.
    # Random azimuth
    randazimuth = np.arccos(2 * np.random.rand(1) - 1)
    # Random elevation
    randelevation = 2 * np.pi * np.random.rand(1)
    # Random axis in cartesian coordinate (this already has norm 1)
    randaxis = [
        np.cos(randelevation) * np.cos(randazimuth),
        np.cos(randelevation) * np.sin(randazimuth),
        np.sin(randelevation)
    ]
    # Random angle
    randangle = 2 * np.pi * np.random.rand(1)
    # Construct axis-angle vector
    randaxisangle = randangle * randaxis

    return np.squeeze(randaxisangle)


def randomized_verts(model,
                     ncomps=12,
                     pose_var=2,
                     trans=None,
                     hand_pose_l=None,
                     hand_pose_r=None,
                     body_pose=None,
                     hand_pose_offset=3,
                     is_cropped=False,
                     center_idx=40,
                     shape_val=2,
                     betas=None,):
    """
    Args:
        model: SMPL+H chumpy model
        smpl_data: 72-dim SMPL pose parameters from CMU and 10-dim shape parameteres from CAESAR
        center_idx: hand root joint on which to center, 25 for left hand
            40 for right
        z_min: min distance to camera in world coordinates
        z_max: max distance to camera in world coordinates
        ncomps: number of principal components used for both hands
        hand_pose: pca coeffs of hand pose
        hand_pose_offset: 3 is hand_pose contains global rotation
            0 if only pca coeffs are provided
    """


    center_idx=1
    # Load smpl

    # Init with zero trans
    model.trans[:] = np.zeros(model.trans.size)

    # Set random shape param
    randshape=betas
    model.betas[:] = randshape
    # if random_shape:
    #     randshape = random.choice(fshapes)
    #     model.betas[:] = randshape
    # else:
    #     randshape = np.zeros(model.betas.shape)
    #model.betas[:] = np.random.uniform(
    #    low=-shape_val, high=shape_val, size=model.betas.shape)

    # Random body pose (except hand)
    randpose = np.zeros(model.pose.size)
    body_idx = 72
    randpose[:72]=body_pose
    randpose[:3]=[0,0,0]


    hand_comps = int(ncomps / 2)
    hand_idx = 66
    if hand_pose_l is not None:
        randpose[hand_idx:hand_idx + hand_comps:] = hand_pose_l[hand_pose_offset:]
        left_rand = hand_pose_l[hand_pose_offset:]
    else:
        left_rand = np.random.uniform(
            low=-pose_var, high=pose_var, size=(hand_comps, ))
        randpose[hand_idx + hand_comps:] = left_rand
        # Alter right hand
    if hand_pose_r is not None:
        randpose[hand_idx + hand_comps:] = hand_pose_r[hand_pose_offset:]
        right_rand = hand_pose_r[hand_pose_offset:]
    else:
        # Alter left hand
        right_rand = np.random.uniform(
            low=-pose_var, high=pose_var, size=(hand_comps, ))
        randpose[hand_idx:hand_idx + hand_comps:] = right_rand

    model.pose[:] = randpose

    # rand_z = random.uniform(z_min, z_max)
    if trans is None:
        trans = np.array(
            [model.J_transformed[center_idx, :].r[i] for i in range(3)])
    # # Offset in z direction
    # trans=camera_trans
    # trans[2]=trans[2]-3
    # trans[2]-=5.5
    # trans[0]-=0.01
        if(is_cropped):
            # trans[2]-=13.8
            trans[2] =2
            # trans[2] = trans[2] + z_min
            trans[1] = trans[1]+.575
            trans[0] = trans[0] - 0.04
        else:
            trans[2] =3.5
            # trans[2] = trans[2] + z_max
            trans[1]=trans[1]-.5
            trans[0] = trans[0] - 0.07
    model.trans[:] = -trans

    new_verts = model.r
  
    hand_pose = right_rand
    
    meta_info = {
        # 'z': rand_z,
        #'trans': (-trans).astype(np.float32),
        'pose': randpose.astype(np.float32),
        'shape': randshape.astype(np.float32),
        # 'mano_pose': hand_pose_l.astype(np.float32)
    }
   
    return new_verts, model, meta_info, trans


def randomized_verts_smplx(model,
                           trans=None,
                           hand_pose_l=None,
                           hand_pose_r=None,
                           body_pose=None,
                           jaw_pose=None,
                           leye_pose=None,
                           reye_pose =None,
                           expression=None,
                           hand_pose_offset=3,
                           camera_trans=None,
                           is_cropped=False,
                           betas=None,
                           gender='female',
                           is_SMPLerX=True):
    """
    Args:
        model: SMPLX chumpy model
        ncomps: number of principal components used for both hands
        hand_pose_l: pca coeff of left hand pose
        hand_pose_r: pca coeff of right hand pose
        hand_pose_offset: 3 is hand_pose contains global rotation
            0 if only pca coeffs are provided
        body_pose: new body pose
        camera_trans: camera translation
        is_cropped: if rendering is cropped
        betas: betas param from smplx
        gender: gender
        is_SMPLerX: if SMPLer-X is used for model fitting (trans differ)
    """

    if trans is None:
        trans = camera_trans
        # print('camera_trans', camera_trans)
        if is_cropped:
            trans[2] = 2
            if is_SMPLerX:
                trans[1] = 0  # .3
            else:
                trans[1] = trans[1] - .2  # .3
        else:
            trans[2] = 3.5
            trans[1] = trans[1] - .6

    use_cuda = True
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    translation = torch.from_numpy(-trans).to(device=device).unsqueeze(dim=0)
    expression = torch.from_numpy(expression).to(device=device)
    body_pose = torch.from_numpy(body_pose).float().to(device=device).unsqueeze(dim=0)
    if isinstance(betas, np.ndarray):
        betas = torch.from_numpy(betas).to(device=device).unsqueeze(dim=0)
    else:
        betas = betas.to(device=device)
    hand_pose_l = torch.from_numpy(hand_pose_l).float().to(device=device)
    hand_pose_r = torch.from_numpy(hand_pose_r).float().to(device=device)
    expression = expression.unsqueeze(dim=0)
    hand_pose_l = hand_pose_l.unsqueeze(dim=0)
    hand_pose_r = hand_pose_r.unsqueeze(dim=0)

    # print('betas', betas.shape)
    # print('body_pose', body_pose.shape)

    output_model = model(betas=betas, body_pose=body_pose, expression=expression, transl=translation,
                         jaw_pose=torch.from_numpy(jaw_pose).float().to(device=device),
                         left_hand_pose=hand_pose_l, right_hand_pose=hand_pose_r,
                         leye_pose=torch.from_numpy(np.array([0.1745329, 0, 0 ])).float().to(device=device),
                         reye_pose=torch.from_numpy(np.array([0.1745329, 0, 0 ])).float().to(device=device),
                         gender=gender, return_verts=True)
    # print(output_model)
    new_verts = output_model.vertices.detach().cpu().numpy().squeeze()

    meta_info = {
        # 'pose': body_pose.detach().cpu().numpy().astype(np.float32).squeeze(),
        # 'betas': betas.detach().cpu().numpy().astype(np.float32), 
        # 'expression': expression.detach().cpu().numpy().astype(np.float32),
        # 'jaw_pose': jaw_pose.astype(np.float32),
        # 'left_hand_pose': hand_pose_l.detach().cpu().numpy().astype(np.float32).squeeze(),
        # 'right_hand_pose': hand_pose_r.detach().cpu().numpy().astype(np.float32).squeeze(),
        'leye_pose': np.array([0.1745329, 0, 0 ]),
        'reye_pose': np.array([0.1745329, 0, 0 ]),
        'gender': gender,
        'transl': trans,
    }

    return new_verts, output_model, meta_info, trans


def load_obj(filename_obj, normalization=True, texture_size=4):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype('float32')

    # load faces
    faces = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype('int32') - 1

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[None, :] / 2

    return vertices, faces
