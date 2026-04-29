import bpy
import bmesh
import mathutils
from mathutils import Vector, Quaternion
from math import exp, sqrt

# --- CONFIGURATION ---
SPLAT_OBJ_NAME = "canGS"  # Ensure this matches your Splat object name exactly
SEARCH_RADIUS = 0.05
MAX_NEIGHBORS = 10

# Output Attribute Names
ATTR_ALBEDO = "Baked_Albedo"
ATTR_NORMAL = "Baked_Normal"
ATTR_ROUGHNESS = "Baked_Roughness"
ATTR_METALLIC = "Baked_Metallic"
# ---------------------

def extract_gaussian_attributes(splat_obj):
    if not splat_obj or splat_obj.type != 'MESH':
        return None

    mesh = splat_obj.data
    count = len(mesh.vertices)
    print(f"Processing {count} Gaussians...")

    positions = [splat_obj.matrix_world @ v.co for v in mesh.vertices]
    rotations = [Quaternion((1,0,0,0))] * count
    scales = [[1,1,1]] * count
    opacities = [1.0] * count
    sh_mags = [0.0] * count
    colors = [Vector((1,1,1))] * count

    attrs = mesh.attributes
    
    # 1. Rotation
    att = attrs.get("rot") or attrs.get("rotation")
    if att:
        for i, d in enumerate(att.data):
            v = d.vector if hasattr(d, 'vector') else d.value
            rotations[i] = Quaternion((v[0], v[1], v[2], v[3]))

    # 2. Scale
    att = attrs.get("scale") or attrs.get("scaling")
    if att:
        for i, d in enumerate(att.data):
            v = d.vector if hasattr(d, 'vector') else d.value
            scales[i] = [exp(v[0]), exp(v[1]), exp(v[2])]

    # 3. Opacity (Sigmoid to keep 0-1)
    att = attrs.get("opacity") or attrs.get("alpha")
    if att:
        for i, d in enumerate(att.data):
            v = d.value
            opacities[i] = 1.0 / (1.0 + exp(-v))

    # 4. Color (SH0 to RGB)
    att_dc0 = attrs.get("f_dc_0")
    if att_dc0:
        for i in range(count):
            r = attrs["f_dc_0"].data[i].value
            g = attrs["f_dc_1"].data[i].value
            b = attrs["f_dc_2"].data[i].value
            colors[i] = Vector((r*0.282 + 0.5, g*0.282 + 0.5, b*0.282 + 0.5))

    # 5. SH Magnitude (Roughness Proxy)
    # We combine the first three f_rest attributes to get a directional vector
    att_sh0 = attrs.get("f_rest_0")
    att_sh1 = attrs.get("f_rest_1")
    att_sh2 = attrs.get("f_rest_2")

    if att_sh0 and att_sh1 and att_sh2:
        for i in range(count):
            # Pull the three floats and make them a Vector
            sh_vec = Vector((
                attrs["f_rest_0"].data[i].value,
                attrs["f_rest_1"].data[i].value,
                attrs["f_rest_2"].data[i].value
            ))
            # Now we can safely call .length
            sh_mags[i] = sh_vec.length
    else:
        print("Warning: f_rest_0/1/2 not found, using flat roughness.")

    return positions, rotations, scales, opacities, colors, sh_mags

def run_pbr_bake():
    mesh_obj = bpy.context.active_object
    splat_obj = bpy.data.objects.get(SPLAT_OBJ_NAME)

    if not mesh_obj or mesh_obj.type != 'MESH':
        print("Select the Retopologized Mesh!")
        return
    
    data = extract_gaussian_attributes(splat_obj)
    if not data: return
    positions, rotations, scales, opacities, colors, sh_mags = data

    print("Building KDTree...")
    kd = mathutils.kdtree.KDTree(len(positions))
    for i, p in enumerate(positions):
        kd.insert(p, i)
    kd.balance()

    mesh = mesh_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # Note: We work in World Space for the search
    bmesh.ops.transform(bm, matrix=mesh_obj.matrix_world, verts=bm.verts)
    
    bm.verts.ensure_lookup_table()

    # --- Updated Attribute Creation Logic ---
    # Using 'FLOAT_COLOR' for better Normal map precision in 4.5+
    l_alb = mesh.attributes.get(ATTR_ALBEDO) or mesh.attributes.new(name=ATTR_ALBEDO, type='FLOAT_COLOR', domain='CORNER')
    l_nrm = mesh.attributes.get(ATTR_NORMAL) or mesh.attributes.new(name=ATTR_NORMAL, type='FLOAT_COLOR', domain='CORNER')
    l_rgh = mesh.attributes.get(ATTR_ROUGHNESS) or mesh.attributes.new(name=ATTR_ROUGHNESS, type='FLOAT_COLOR', domain='CORNER')
    l_met = mesh.attributes.get(ATTR_METALLIC) or mesh.attributes.new(name=ATTR_METALLIC, type='FLOAT_COLOR', domain='CORNER')

    print("Baking Vertex Data...")

    for loop in mesh.loops:
        vert = bm.verts[loop.vertex_index]
        v_pos = vert.co
        
        found = kd.find_n(v_pos, MAX_NEIGHBORS)
        
        w_sum = 0.0
        col_acc = Vector((0,0,0))
        nrm_acc = Vector((0,0,0))
        rgh_acc = 0.0
        met_acc = 0.0
        
        for (p, idx, dist) in found:
            w = 1.0 / (dist + 0.0001)
            w_sum += w
            
            # --- Albedo ---
            col_acc += colors[idx] * w
            
            # --- Metallic ---
            met_acc += (1.0 - opacities[idx]) * w
            
            # --- Roughness ---
            sh_norm = min(sh_mags[idx] / 2.0, 1.0)
            rgh_acc += (1.0 - sh_norm) * w
            
            # --- Normal Logic ---
            sc = scales[idx]
            rot = rotations[idx]
            min_idx = sc.index(min(sc))
            axis = [Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1))][min_idx]
            
            n = rot @ axis
            # Orient normal toward the outside of our retopo mesh
            if n.dot(vert.normal) < 0:
                n = -n
            nrm_acc += n * w

        # Finalize and write
        if w_sum > 0:
            f_col = col_acc / w_sum
            f_met = met_acc / w_sum
            f_rgh = rgh_acc / w_sum
            f_nrm = (nrm_acc / w_sum).normalized()

            l_alb.data[loop.index].color = (f_col.x, f_col.y, f_col.z, 1.0)
            l_met.data[loop.index].color = (f_met, f_met, f_met, 1.0)
            l_rgh.data[loop.index].color = (f_rgh, f_rgh, f_rgh, 1.0)
            
            n_map = (f_nrm + Vector((1,1,1))) * 0.5
            l_nrm.data[loop.index].color = (n_map.x, n_map.y, n_map.z, 1.0)

    bm.free()
    print("Baking Complete! Check your Vertex Color layers.")

run_pbr_bake()
