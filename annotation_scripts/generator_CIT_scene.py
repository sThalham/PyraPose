import os,sys
import bpy
import numpy as np
from random import randint
from random import random
from numpy.random import normal
from random import uniform
from random import choice as Rchoice
from random import sample
import cv2
import yaml
import itertools
from math import radians,degrees,tan,cos
from numpy.linalg import inv

# visible vertices
import bmesh
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

pcd_header = '''VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 640
HEIGHT 480
VIEWPOINT 0 0 0 1 0 0 0
POINTS 307200
DATA ascii
'''

#Height : 480 for kinect, 512 for ensenso
#Points : 307200 for kinect, 327680 for ensenso
# tless: 720x540

def getVerticesVisible(obj):

    scene = bpy.context.scene
    cam = bpy.data.objects['cam_R']
    mesh = obj.data
    mat_world = obj.matrix_world
    cs, ce = cam.data.clip_start, cam.data.clip_end
    
    # edit mode to edit meshes
    scene.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    #bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bm = bmesh.from_edit_mesh(mesh)

    limit = 0.1
    #vertices = [mat_world * v.co for v in obj.data.vertices]
    #for i, v in enumerate( vertices ):
    for v in bm.verts:
        v.select = True   # need it for vertices hidden in last iteration
        vec = mat_world * v.co
        co_ndc = world_to_camera_view(scene, cam, vec)
        #co_ndc = world_to_camera_view(scene, cam, v)
        #check wether point is inside frustum
        #if (0.0 < co_ndc.x < 1.0 and
        #    0.0 < co_ndc.y < 1.0 and
        #     cs < co_ndc.z <  ce):
            # ray_cast if point is visible
        #    results, obj, mat, location, normal = scene.ray_cast(cam.location, (vec - cam.location).normalized() )
        #    if location and (vec - location).length < limit:
        #        v.select = False
        #    else:
        #        v.select = True
        #else:
        #    v.select = True
            
    # limit selection to visible from certain camera view
    #bpy.ops.mesh.hide(unselected=False)
    #bmesh.update_edit_mesh(mesh, False, False)  
    bpy.ops.object.mode_set(mode='OBJECT')    
     

def getVisibleBoundingBox(objectPassIndex):

    S = bpy.context.scene
    width  = int( S.render.resolution_x * S.render.resolution_percentage / 100 )
    height = int( S.render.resolution_y * S.render.resolution_percentage / 100 )
    depth  = 4

    pixels = np.array( bpy.data.images['Render Result'].pixels[:] ).reshape( [height, width, depth] )
    # Keep only one value for each pixel (white pixels have 1 in all RGBA channels anyway), thus converting the image to black and white
    pixels = np.array( [ [ pixel[0] for pixel in row ] for row in pixels ] )

    bbox = np.argwhere( pixels == objectPassIndex )
    (ystart, xstart), (ystop, xstop) = bbox.min(0), bbox.max(0) + 1
    bb = (xstart, xstart, height - ystart, height - ystop)
    return bb, bbox


# 23.4.2018
# render image 350 of object 23
# cam_K: [1076.74064739, 0.0, 215.98264967, 0.0, 1075.17825536, 204.59181836, 0.0, 0.0, 1.0]
# depth_scale: 0.1
# elev: 45
# mode: 0

# cam_R_m2c: [0.62268218, -0.78164004, -0.03612308, -0.56354950, -0.41595975, -0.71371609, 0.54284357, 0.46477486, -0.69950372]
# cam_t_m2c: [-9.10674129, -2.47862668, 634.81667094]
# obj_bb: [120, 121, 197, 190]
# obj_id: 23

# f = 580
# b = 0.0075

#base_dir = "/home/sthalham/data/LINEMOD/models_stl_red_13"
base_dir = "/home/stefan/data/Meshes/CIT_color"
back_img_path = '/home/stefan/data/datasets/cocoval2017'
all_backgrounds = os.listdir(back_img_path)
total_set = 1 #10000 set of scenes, each set has identical objects with varied poses to anchor pose (+-15)
pair_set = 3 #number of pair scene for each set, 10
sample_dir = '/home/stefan/data/renderings/CIT_render' #directory for temporary files (cam_L, cam_R, masks..~)
target_dir = '/home/stefan/data/renderings/CIT_render/patches'
index=0
isfile=True
while isfile:
    prefix='{:08}_'.format(index)
    if(os.path.exists(os.path.join(target_dir,prefix+'gt.yaml'))):
        index+=1
    else:
        isfile=False

if not(os.path.exists(target_dir+"/mask")):
    os.makedirs(target_dir+"/mask")
    
if not(os.path.exists(target_dir+"/rgb")):
    os.makedirs(target_dir+"/rgb")


model_file=[]
model_solo=[]
for root, dirs, files in os.walk(base_dir):
    for file in sorted(files):
        if file.endswith(".ply"):
             temp_fn =os.path.join(root, file)
             model_file.append(temp_fn)
             model_solo.append(file)
             #print(len(model_file),temp_fn)
             
dis_objects = ['dis1_1', 'dis1_2', 'dis1_3', 'dis2_1', 'dis2_2', 'dis2_3']


# FOR BACKGROUND OBJECTS  

for num_set in np.arange(total_set):
    bpy.ops.object.select_all(action='DESELECT')
    scene = bpy.context.scene
    scene.objects.active = bpy.data.objects["template"]
    for obj in scene.objects:
        if obj.type == 'MESH':
            if obj.name == 'template':
                obj.select = False
            elif obj.name[0:5] == 'Plane':
                obj.select = False
            elif obj.name[0:3] == 'dis':
                obj.select = False
            elif obj.name == 'InvisibleCube':
                obj.select = False
            else:
                obj.select = True
        if obj.name[0:6] == 'light_':
            obj.select = True

    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')
    obj_object = bpy.data.objects["template"]
    obj_object.pass_index = 1
    mat = obj_object.active_material
    
    # memory is leaking
    # that should fix it
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

    # sample objects
    num_object = np.random.randint(low=4, high=12)
    num_distractor = np.random.randint(low=2, high=4)
    
    ObjDraw = np.random.choice(np.arange(len(model_solo)), size=num_object, replace=True)
    DisDraw = np.random.choice(np.arange(len(dis_objects)), size=num_distractor, replace=False)
    
    object_label =[]
    anchor_pose = np.zeros(((num_distractor + num_object+1),6)) 
    object_idx = 0
    
    print(num_distractor)
    print(num_object)
    print(anchor_pose.shape)
    
    for file_idx in ObjDraw:
        file_model = model_file[file_idx]
        solo_model = model_solo[file_idx]
        imported_object = bpy.ops.import_mesh.ply(filepath=file_model, filter_glob="*.ply", files=[{"name":solo_model, "name":solo_model}], directory=root)
        object_label.append(object_idx)
        obj_object = bpy.context.selected_objects[0]
        mat_obj = mat.copy()
        tree_mesh = mat_obj.node_tree
        nodes_mesh = tree_mesh.nodes
        nodes_mesh['Attribute'].color = (0.5, 0.5, 0.5)
        
        obj_object.active_material = mat_obj
        obj_object.pass_index = object_idx +2 # don't add?
        print(object_idx)
        anchor_pose[object_idx+1,0] = random()*0.5 - 0.25
        anchor_pose[object_idx+1, 1] = random()*0.5 - 0.25
        anchor_pose[object_idx+1,2] = 0.3 + random()*0.5
        anchor_pose[object_idx+1,3] = radians(random()*360.0)
        anchor_pose[object_idx+1,4] = radians(random()*360.0)
        anchor_pose[object_idx+1,5] = radians(random()*360.0)
        print(anchor_pose[object_idx+1, :])
        object_idx += 1
        print(obj_object, obj_object.pass_index, object_idx)
        
    print("FOREGROUND IMPORTED")
    # Background objects
    for name in dis_objects:
        obj_object = bpy.data.objects[name]
        obj_object.hide_render = True
    
    for file_idx in DisDraw:
        object_label.append(object_idx + num_object)
        obj_object = bpy.data.objects[dis_objects[file_idx]]
        obj_object.hide_render = False
        obj_object.active_material = mat
        obj_object.pass_index = object_idx+2
        anchor_pose[object_idx+1,0] = random()*0.5 - 0.25
        anchor_pose[object_idx+1, 1] = random()*0.5 - 0.25
        anchor_pose[object_idx+1,2] = 0.3 + random()*0.5
        anchor_pose[object_idx+1,3] = radians(random()*360.0)
        anchor_pose[object_idx+1,4] = radians(random()*360.0)
        anchor_pose[object_idx+1,5] = radians(random()*360.0)
        object_idx += 1
        print(obj_object, obj_object.pass_index, object_idx)
 
    # FOR BACKGROUND OBJECTS 
    print("BACKGROUND IMPORTED")
    
    #bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
  
    #Set object physics
    scene = bpy.context.scene
    scene.objects.active = bpy.data.objects["template"]
    for obj in scene.objects:
        if obj.type == 'MESH':
            #print("objects parsed: ", obj)
            if obj.name == 'template':
                obj.select = False          
            elif obj.name[0:5] == 'Plane':
                obj.select = False
            elif obj.name == 'InvisibleCube':
                obj.select = False
            else:
                obj.select = True
    
    
    print("BACKGROUND objects set to inactive for physics")

    bpy.ops.rigidbody.object_settings_copy()
    
  
    #Define Object position&rotation
    
    for iii in np.arange(pair_set):

        tree = bpy.context.scene.node_tree
        nodes = tree.nodes
        ind_obj_counter = 0
        scene.frame_set(0)
        for obj in scene.objects:
            if obj.type == 'MESH':
                obj_object= bpy.data.objects[obj.name]
                
            if obj.name[0:5] == 'Plane':
                obj_object= bpy.data.objects[obj.name]
                img_choice = Rchoice(all_backgrounds)
                img_path = os.path.join(back_img_path, img_choice)
                mat_now = obj_object.active_material
                tree_mesh = mat_now.node_tree
                nodes_mesh = tree_mesh.nodes
                image = bpy.data.images.load(filepath = img_path)
                nodes_mesh['Image Texture'].image = image
            
            if obj_object.pass_index > 1:
                idx = obj_object.pass_index -1
                print(obj_object, obj_object.pass_index, idx)
                obj_object.location.x=anchor_pose[idx,0]
                obj_object.location.y=anchor_pose[idx,1]
                obj_object.location.z=anchor_pose[idx,2]
                obj_object.rotation_euler.x= anchor_pose[idx,3]
                obj_object.rotation_euler.y= anchor_pose[idx,4]
                obj_object.rotation_euler.z= anchor_pose[idx,5]
                 
                # assign different color
                rand_color = (random(), random(), random())
                obj_object.active_material.diffuse_color = rand_color
                if obj.name[:3] == 'dis':
                #if obj_object.pass_index > (num_object+1):
                    obj_object.pass_index = 0
                    print('pass_idx to 0')
              
            if obj.name == 'InvisibleCube':
                obj_object.rotation_euler.x=radians(random()*60.0 + 15.0) #0~90
                obj_object.rotation_euler.y=radians(random()*5.0 - 10.0) #-45-45
                obj_object.rotation_euler.z=radians(random()*360.0) #0-360

            if obj.type == 'CAMERA' and  obj.name=='cam_L':
                obj_object = bpy.data.objects[obj.name]
                obj_object.location.z = random()*0.25+0.45  #1.0-2.5
                
        print("start running physics")    

        #Run physics
        count = 60
        scene.frame_start = 1
        scene.frame_end = count + 1
        print("Start physics")
        for f in range(1,scene.frame_end+1):
            print("scene iteration: ", f, "/60")
            scene.frame_set(f)
            if f <= 1:
                continue
            
        print("pyshics ran")
                
        tree = bpy.context.scene.node_tree
        nodes = tree.nodes
        
        print("render images")
        #When Rander cam_L, render mask together

        prefix='{:08}_'.format(index)
        index+=1
        
        scene.cycles.samples=20
        
        # randomize lights and position
        light_numbers = list(range(1,3))
        light_number = np.bincount(light_numbers)
        lights = np.random.choice(np.arange(len(light_number)), 1, p=light_number / len(light_numbers), replace=False)
        lights = np.asscalar(lights)
    
        for lamp in range(1, lights):
            lamp_types = ['POINT', 'SPOT', 'HEMI', 'AREA']
            lamp_type = Rchoice(lamp_types)
            print('lamp_type: ', lamp_type)
            lamp_name = 'light_' + str(lamp)
            print('lamp_name: ', lamp_name)
            lamp_position = ((random()*5.0-2.5), (random()*2.5-1.25), (random()*1.5+1.5))
            print('lamp_position: ', lamp_position)
            lamp_data = bpy.data.lamps.new(name=lamp_name, type=lamp_type)
            lamp_object = bpy.data.objects.new(name=lamp_name, object_data=lamp_data)
            scene.objects.link(lamp_object)
            lamp_object.location = lamp_position
            lamp_object.select = True
            #scene.objects.active = lamp_object
        

        maskfile = os.path.join(target_dir+'/mask' , 'mask.png')  # correspondence mask
        rgbfile= os.path.join(target_dir+"/rgb", prefix+'rgb.png')   # rgb image

        for ob in scene.objects:
            if ob.type == 'CAMERA':          
                if ob.name=='cam_L': #ob.name =='mask':
                    #Render image and Mask
                    bpy.context.scene.camera = ob
                    print('Set camera %s for IR' % ob.name )
                    file_L = os.path.join(sample_dir , ob.name )
                    
                    auto_file = os.path.join(sample_dir, ob.name+'0061.png')  
                    node= nodes['maskout']
                    node.file_slots[0].path = ob.name
                    node_mix = nodes['ColorRamp']
                    link_mask= tree.links.new(node_mix.outputs["Image"], node.inputs[0])
                    node.base_path=sample_dir                 

                    auto_file_rgb = os.path.join(sample_dir+'/rgb/', ob.name+'0061.png')
                    node= nodes['rgbimage.001']
                    node.file_slots[0].path = ob.name
                    node_mix = nodes['Render Layers']
                    link_rgb = tree.links.new(node_mix.outputs["DiffCol"], node.inputs[0])
                    node.base_path=sample_dir+'/rgb/'
                  
                    scene.render.filepath = file_L
                    bpy.ops.render.render( write_still=True )
                    tree.links.remove(link_mask)
                    tree.links.remove(link_rgb)
                  
                    os.rename(auto_file, maskfile)
                    os.rename(auto_file_rgb, rgbfile)
            

        mask = cv2.imread(maskfile)

        minmax_vu = np.zeros((num_object,4),dtype=np.int) #min v, min u, max v, max u
        label_vu = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.int8) #min v, min u, max v, max u
        colors = np.zeros((num_object,3),dtype=mask.dtype)
        print(colors.shape)

        n_label=0

        color_index=np.array([  [  0, 0,   0],
                        [  0, 100,   0],
                        [  0, 139,   0],
                        [  0, 167,   0],
                        [  0, 190,   0],
                        [  0, 210,   0],
                        [  0, 228,   0],
                        [  0, 244,   0],
                        [  0, 252,  50],
                        [  0, 236, 112],
                        [  0, 220, 147],
                        [  0, 201, 173],
                        [  0, 179, 196],
                        [  0, 154, 215],
                        [  0, 122, 232],
                        [  0,  72, 248],
                        [ 72,   0, 248],
                        [122,   0, 232],
                        [154,   0, 215],
                        [179,   0, 196],
                        [201,   0, 173],
                        [220,   0, 147],
                        [236,   0, 112],
                        [252,   0,  50],
                        [255,  87,  87],
                        [255, 131, 131],
                        [255, 161, 161],
                        [255, 185, 185],
                        [255, 206, 206],
                        [255, 224, 224],
                        [255, 240, 240],
                        [255, 255, 255]])


        for v in np.arange(mask.shape[0]):
            for u in np.arange(mask.shape[1]):
                has_color = False
                if not(mask[v,u,0] ==0 and mask[v,u,1] ==0 and mask[v,u,2] ==0):
                    for ob_index in np.arange(n_label):
                        if colors[ob_index,0]== mask[v,u,0] and colors[ob_index,1]== mask[v,u,1] and colors[ob_index,2]== mask[v,u,2]:
                            has_color = True
                            minmax_vu[ob_index,0] = min(minmax_vu[ob_index,0], v)
                            minmax_vu[ob_index,1] = min(minmax_vu[ob_index,1], u)
                            minmax_vu[ob_index,2] = max(minmax_vu[ob_index,2], v)
                            minmax_vu[ob_index,3] = max(minmax_vu[ob_index,3], u)
                            label_vu[v,u]=ob_index+1
                            continue
                    if has_color ==False: #new label
                        colors[n_label] = mask[v,u]
                        label_vu[v,u]=n_label+1 #identical to object_index in blender
                        minmax_vu[n_label,0] = v
                        minmax_vu[n_label,1] = u
                        minmax_vu[n_label,2] = v
                        minmax_vu[n_label,3] = u
                        n_label=n_label+1
                else:
                    label_vu[v,u]=0

        bbox_refined = mask
        color_map=np.zeros(n_label)

        for k in np.arange(n_label):
            for i in np.arange(color_index.shape[0]):
                if(color_index[i,0] == colors[k,0] and color_index[i,1] == colors[k,1] and color_index[i,2] == colors[k,2] ):
                    color_map[k]=i
                    continue

        object_no=[]
        refined=[]

        for ob_index in np.arange(n_label): #np.arange(n_label):
            min_v=minmax_vu[ob_index,0]
            min_u=minmax_vu[ob_index,1]
            max_v=minmax_vu[ob_index,2]
            max_u=minmax_vu[ob_index,3]
            bbox = label_vu[min_v:max_v,min_u:max_u]
            bbox=bbox.reshape(-1)
            counts = np.bincount(bbox)
            #print('counts: ', counts)
            #print(colors[ob_index])
            if(counts.shape[0]>1):
                if(np.argmax(counts[1:]) ==(ob_index)): #(mask.shape[0],mask.shape[1]
                #if(min_v>30 and min_u>30 and max_v < (mask.shape[0]-30) and max_u < (mask.shape[1]-30) ):
                #cv2.rectangle(bbox_refined,(min_u,min_v),(max_u,max_v),(0,255,0),1)
                    refined.append(ob_index)
                    object_no.append(color_map[ob_index])
                    #print(color_map[ob_index])

        #cv2.imwrite(os.path.join(sample_dir,'bbox_refined.png'),bbox_refined)
        bbox_refined = minmax_vu[refined]
        poses =np.zeros((len(object_no),4,4),dtype=np.float)
        names = ['a'] * len(object_no)
        #visibilities =np.zeros((len(object_no)),dtype=np.float)
        camera_rot =np.zeros((4,4),dtype=np.float)
        for obj in scene.objects:
            if obj.type == 'MESH':
                if obj.pass_index in object_no:
                    idx = object_no.index(obj.pass_index)
                    poses[idx]=obj.matrix_world
                    img_name = obj.name + '.png'
                    names[idx] = os.path.join(sample_dir, img_name)
                if obj.name=='InvisibleCube':
                    camera_rot[:,:] = obj.matrix_world
                    camera_rot = camera_rot[:3,:3] #only rotation (z was recorded seprately)
                    init_rot = np.zeros((3,3))
                    init_rot[0,0]=1
                    init_rot[1,1]=-1
                    init_rot[2,2]=-1
                    fin_rot =np.matmul(camera_rot,init_rot)
                    fin_rot = inv(fin_rot)
                    world_rot=np.zeros((4,4))
                    world_rot[:3,:3] = fin_rot
                    world_rot[3,3]=1
            if obj.type == 'CAMERA' and  obj.name=='cam_L':
                obj_object = bpy.data.objects[obj.name]
                camera_z = obj_object.location.z
                #camera_ext[:,:] = obj_object.matrix_world
                #camera_ext = camera_ext.reshape(-1)

        np.save(target_dir+"/mask/"+prefix+"mask.npy",label_vu)
        cam_trans = -np.matmul(camera_rot[:3, :3],np.array([0,0,camera_z]))
        world_trans =np.zeros((4,4))
        world_trans[0,0]=1
        world_trans[1,1]=1
        world_trans[2,2]=1
        world_trans[3,3]=1
        world_trans[:3,3] = cam_trans

        masksT = []
        boxesT = []
        #camOrientation = np.array(bpy.data.objects['cam_L'].matrix_world).reshape(-1)
        camOrientation =np.zeros((4,4),dtype=np.float)
        camOrientation[3,3]=1.0
        camOrientation[:3,3] = cam_trans
        camOrientation[:3,:3] = world_rot[:3,:3]
        camOrientation = np.linalg.inv(camOrientation)
        with open(os.path.join(target_dir,prefix+'gt.yaml'),'w') as f:
            camOri={'camera_rot':camOrientation.tolist()}
            yaml.dump(camOri,f)
            for i in np.arange(len(object_no)):
                pose = poses[i]
                pose = np.matmul(world_trans,pose)
                pose = np.matmul(world_rot,pose)
                pose_list=pose.reshape(-1)
                id = int(object_label[int(object_no[i]-2)])
                mask_id = int(refined[i]+1)
                gt={int(i):{'bbox':bbox_refined[i].tolist(),'class_id':id,'mask_id':mask_id,'pose':pose_list.tolist()}} #,'camera_z':camera_z,'camera_rot':camera_rot.tolist()
                yaml.dump(gt,f)
    
    print("Iterations done: ", num_set, "/", total_set, " done ")
    
    
bpy.ops.object.select_all(action='DESELECT')
scene = bpy.context.scene
scene.objects.active = bpy.data.objects["template"]
for obj in scene.objects:
    if obj.type == 'MESH':
        if obj.name[:2] == 'template':
            obj.select = False
        elif obj.name[0:5] == 'Plane':
            obj.select = False
        elif obj.name[0:3] == 'dis':
            obj.select = False
        elif obj.name == 'InvisibleCube':
            obj.select = False
        else:
            obj.select = True
    if obj.name[0:6] == 'light_':
        obj.select = True

bpy.ops.object.delete()    

print("Relax, all good and finished")
