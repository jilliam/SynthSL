import os
import uuid
import numpy as np

import bpy


def get_all_backgrounds(folder='/sequoia/data2/gvarol/datasets/LSUN/data/img'):
    train_list_path = os.path.join(folder, 'train_img.txt')
    with open(train_list_path) as f:
        lines = f.readlines()
    bg_names = [os.path.join(folder, line.strip()) for line in lines]
    return bg_names


def set_cycle_nodes(scene, bg_name, segm_path, depth_path, normal_path, gtflow_path):
    # Get node tree
    scene.use_nodes = True
    # scene.render.layers['RenderLayer'].use_pass_material_index = True
    scene.view_layers["ViewLayer"].use_pass_material_index = True

    # scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.view_layers["ViewLayer"].use_pass_emit = True
    scn = bpy.context.scene
    scn.cycles.film_transparent = True
    scn.view_layers["ViewLayer"].use_pass_z = True
    scn.view_layers["ViewLayer"].use_pass_vector = True
    scn.view_layers["ViewLayer"].use_pass_normal = True
    # scn.view_layers["ViewLayer"].use_motion_blur = True

    # scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.film_transparent = 1 # 0 for "SKY"; 1 for "TRANSPARENT
    node_tree = scene.node_tree

    # Remove existing nodes
    for n in node_tree.nodes:
        node_tree.nodes.remove(n)

    # # Create node for the final output
    # composite_out = node_tree.nodes.new('CompositorNodeComposite')

    # Get background
    bg_node = node_tree.nodes.new(type="CompositorNodeImage")

    if bg_name is not None:
        bg_img = bpy.data.images.load(bg_name)
    bg_node.image = bg_img
    bg_node.location = -400, 200

    # Scale background by croping
    scale_node = node_tree.nodes.new(type="CompositorNodeScale")
    scale_node.space = "RENDER_SIZE"
    scale_node.frame_method = "CROP"
    scale_node.location = -200, 200
    node_tree.links.new(bg_node.outputs[0], scale_node.inputs[0])

    # Get rendering
    render_node = node_tree.nodes.new(type="CompositorNodeRLayers")
    render_node.location = -400, -200

    # Save depth
    depth_node =  node_tree.nodes.new('CompositorNodeOutputFile')
    depth_node.format.file_format = 'OPEN_EXR'
    depth_node.base_path = os.path.dirname(depth_path)
    depth_node.file_slots[0].path = os.path.basename(depth_path)
    depth_node.location = 200, 0
    # print('depth_node.inputs', depth_node.inputs)
    node_tree.links.new(render_node.outputs["Depth"], depth_node.inputs[0])
    # node_tree.links.new(render_node.outputs['Z'], depth_node.inputs[0])
    # node_tree.links.new(render_node.outputs["Depth"], depth_node.inputs['Image'])

    # Save normal
    normal_node =  node_tree.nodes.new('CompositorNodeOutputFile')
    normal_node.format.file_format = 'OPEN_EXR'
    normal_node.base_path = os.path.dirname(normal_path)
    normal_node.file_slots[0].path = os.path.basename(normal_path)
    normal_node.location = 200, 0
    node_tree.links.new(render_node.outputs["Normal"], normal_node.inputs[0])

    # vecblur = node_tree.nodes.new('CompositorNodeVecBlur')
    # vecblur.samples = 64
    # vblur_factor_Miu = 0.03
    # vblur_factor_Std = 0.03
    # vblur_factor = np.random.normal(vblur_factor_Miu, vblur_factor_Std)
    # vecblur.factor = vblur_factor
    # finalblur = node_tree.nodes.new('CompositorNodeBlur')
    # node_tree.links.new(render_node.outputs['Image'], vecblur.inputs[0])
    # node_tree.links.new(render_node.outputs['Vector'], vecblur.inputs[2])
    # # node_tree.links.new(finalblur.outputs[0], composite_out.inputs[0])

    # Save ground truth flow
    gtflow_out = node_tree.nodes.new('CompositorNodeOutputFile')
    gtflow_out.format.file_format = 'OPEN_EXR'
    gtflow_out.base_path = os.path.dirname(gtflow_path)
    gtflow_out.file_slots[0].path = os.path.basename(gtflow_path)
    # gtflow_out.location = 200, 0
    node_tree.links.new(render_node.outputs['Vector'], gtflow_out.inputs[0])
    # node_tree.links.new(render_node.outputs['Speed'], gtflow_out.inputs[0])

    # Overlay background image and rendering
    alpha_node = node_tree.nodes.new(type="CompositorNodeAlphaOver")
    alpha_node.location = 0, 200
    node_tree.links.new(scale_node.outputs[0], alpha_node.inputs[1])
    node_tree.links.new(render_node.outputs[0], alpha_node.inputs[2])

    comp_node = node_tree.nodes.new(type="CompositorNodeComposite")
    comp_node.location = 200, 200
    node_tree.links.new(alpha_node.outputs[0], comp_node.inputs[0])

    # Add segmentation node
    scale_node = node_tree.nodes.new(type='CompositorNodeMapRange')
    scale_node.location = 0, -200
    scale_node.inputs[1].default_value = 0
    scale_node.inputs[2].default_value = 255
    # left_handbase_mask.index = 21 # left_finger_mask.index = 23
    segm_view = node_tree.nodes.new(type="CompositorNodeOutputFile")
    # segm_view.location = 200, -200
    segm_view.format.file_format = 'PNG'
    segm_view.base_path = segm_path
    temp_filename = uuid.uuid4().hex
    segm_view.file_slots[0].path = temp_filename
    node_tree.links.new(render_node.outputs['IndexMA'], scale_node.inputs[0])
    node_tree.links.new(scale_node.outputs[0], segm_view.inputs[0])

    # segm_out = node_tree.nodes.new("CompositorNodeOutputFile")
    # segm_out.location = 200, -200
    # segm_out.format.file_format = "OPEN_EXR"
    # temp_filename = uuid.uuid4().hex
    # segm_out.file_slots[0].path = temp_filename
    # node_tree.links.new(render_node.outputs["IndexMA"], segm_out.inputs[0])

    tmp_segm_file = os.path.join(segm_path, '{}0001.png'.format(temp_filename))
    return tmp_segm_file
