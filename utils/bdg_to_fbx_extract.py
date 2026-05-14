from pathlib import Path
import struct, math, shutil, zipfile, json, io, collections, hashlib
from PIL import Image
import sys, argparse, re
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
from decode_cmpr import decode_cmpr
from wii_tex_decode import decode_rgb565, decode_i8



def clean_windows_folder_arg(value):
    """Normalize folder args passed by .bat files on Windows.

    Windows command-line parsing can leave a stray trailing quote when a quoted
    %~dp0 path ends in a backslash. Accept and clean that instead of crashing.
    """
    value = str(value).strip()
    while value and value[-1] in ('\"', "'"):
        value = value[:-1].rstrip()
    while value and value[0] in ('\"', "'"):
        value = value[1:].lstrip()
    return value or '.'

def find_case_insensitive(path: Path, name: str):
    p = path / name
    if p.exists():
        return p
    lname = name.lower()
    for q in path.iterdir():
        if q.name.lower() == lname:
            return q
    return None

def find_inputs(root: Path):
    shapes = sorted([p for p in root.iterdir() if p.is_file() and p.name.lower().endswith('_shapes.bdg')])
    if not shapes:
        raise SystemExit('No *_Shapes.BDG found beside Extract.bat. Put one kaiju Shapes BDG in this folder.')
    if len(shapes) > 1:
        names='\n  '.join(p.name for p in shapes)
        raise SystemExit('More than one *_Shapes.BDG found. Keep one kaiju set in the folder at a time:\n  '+names)
    shape=shapes[0]
    base=re.sub(r'_Shapes\.BDG$', '', shape.name, flags=re.I)
    anim=find_case_insensitive(root, base + '.BDG')
    if anim is None:
        raise SystemExit(f'Missing matching animation/control BDG: {base}.BDG')
    pvms=sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower()=='.pvm' and p.name.lower().startswith(base.lower())])
    return base, shape, anim, pvms

def parse_args():
    ap=argparse.ArgumentParser(description='Extract supported Pipeworks/Spigot BDG kaiju assets to FBX. Current proven profile: Godzilla2K/Godzilla_Shapes.BDG.')
    ap.add_argument('folder', nargs='?', default='.', help='Folder containing one kaiju *_Shapes.BDG, matching *.BDG, and optional *_Root.pvm')
    ap.add_argument('--force', action='store_true', help='Overwrite existing output folder')
    return ap.parse_args()

ARGS=parse_args()
ROOT=Path(clean_windows_folder_arg(ARGS.folder)).resolve()
MONSTER_BASE, SRC, ANIM_SRC, PVM_FILES = find_inputs(ROOT)
D = SRC.read_bytes()
if b'GODZILLA2K_SKELETON' not in D or b'GODZILLA2K_MESH' not in D:
    raise SystemExit('This attempt currently contains the proven Godzilla2K profile only. Codex should generalize the profile scanner using the note. This Shapes.BDG does not match GODZILLA2K.')
OUT = ROOT / f'{MONSTER_BASE}-Kaiju-Extracted'
TEX = OUT/'textures'
if OUT.exists():
    if not ARGS.force:
        raise SystemExit(f'Output folder already exists: {OUT}. Delete it or run with --force.')
    shutil.rmtree(OUT)
TEX.mkdir(parents=True)

# ------------------------- Texture decode -------------------------
DATA_BASE = 0xCE00
texture_specs = {
    'Godzilla2k_G2_512_C.png': ('CMPR', 0x000000, 512, 512),
    'Godzilla2k_G2_512_B_raw.png': ('RGB565', 0x02AAC0, 512, 512),
    'Godzilla2k_G2_512_S.png': ('CMPR', 0x12D760, 512, 512),
    'Godzilla2k_G2_256_M.png': ('I8', 0x158220, 256, 256),
}
for name,(fmt,rel,w,h) in texture_specs.items():
    raw = D[DATA_BASE+rel:]
    if fmt == 'CMPR': img = decode_cmpr(raw[:w*h//2], w, h)
    elif fmt == 'RGB565': img = decode_rgb565(raw[:w*h*2], w, h)
    elif fmt == 'I8': img = decode_i8(raw[:w*h], w, h)
    img.save(TEX/name)
# Blender-friendly normal derived from raw B map.
raw_n = Image.open(TEX/'Godzilla2k_G2_512_B_raw.png').convert('RGBA')
normal = Image.new('RGBA', raw_n.size)
sp=raw_n.load(); dp=normal.load()
for y in range(raw_n.height):
    for x in range(raw_n.width):
        r,g,b,a=sp[x,y]
        nx = r/255.0*2.0-1.0
        ny = g/255.0*2.0-1.0
        l2=min(1.0,nx*nx+ny*ny)
        nz=math.sqrt(max(0.0,1.0-l2))
        dp[x,y]=(int((nx*.5+.5)*255+.5), int((-ny*.5+.5)*255+.5), int((nz*.5+.5)*255+.5), a)
normal.save(TEX/'Godzilla2k_G2_512_N.png')

# ------------------------- True skeleton decode -------------------------
# String table uses little-endian offsets from 0x400.
str_count = struct.unpack('<I', D[0x400:0x404])[0]
ptrs = struct.unpack('<'+'I'*str_count, D[0x404:0x404+4*str_count])
strings=[]
for p in ptrs:
    off=0x400+p; end=D.find(b'\0',off)
    strings.append(D[off:end].decode('latin1', errors='replace'))

SKEL_BASE = 0x0EE0
SKEL_ROOT = 0x0F20
skeleton={}
def parse_skel_record(off):
    idx, parent, nchild, name_idx = struct.unpack('>4i', D[off:off+16])
    q = struct.unpack('>4f', D[off+16:off+32]) # x,y,z,w
    t = struct.unpack('>3f', D[off+32:off+44]) # local translation
    child_rels = struct.unpack('>'+('I'*nchild), D[off+48:off+48+4*nchild]) if nchild else ()
    return {
        'idx':idx, 'parent':parent, 'nchild':nchild,
        'name_idx':name_idx, 'name':strings[name_idx], 'q':q, 't':t,
        'children':[SKEL_BASE+c for c in child_rels], 'off':off,
    }
def walk(off):
    r=parse_skel_record(off); skeleton[r['idx']]=r
    for c in r['children']:
        walk(c)
walk(SKEL_ROOT)
BONE_COUNT = max(skeleton)+1
bone_names=[skeleton[i]['name'] for i in range(BONE_COUNT)]
parent={i:skeleton[i]['parent'] for i in range(BONE_COUNT)}

# Quaternion/matrix helpers.
def qmat(q):
    x,y,z,w=q
    n=x*x+y*y+z*z+w*w
    if n < 1e-12: return [[1,0,0],[0,1,0],[0,0,1]]
    s=2.0/n
    xx,yy,zz=x*x*s,y*y*s,z*z*s
    xy,xz,yz=x*y*s,x*z*s,y*z*s
    wx,wy,wz=w*x*s,w*y*s,w*z*s
    return [[1-yy-zz, xy-wz, xz+wy], [xy+wz, 1-xx-zz, yz-wx], [xz-wy, yz+wx, 1-xx-yy]]

def matmul(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def local_matrix(i):
    r=skeleton[i]
    R=qmat(r['q']); x,y,z=r['t']
    # FBX-style row-major matrix with translation in final row, using T*R equivalent for row vectors.
    return [[R[0][0],R[0][1],R[0][2],0], [R[1][0],R[1][1],R[1][2],0], [R[2][0],R[2][1],R[2][2],0], [x,y,z,1]]
# For global joint positions, use column-vector math as in validation; for FBX bind matrices flatten row-major.
def col_local_matrix(i):
    r=skeleton[i]
    R=qmat(r['q']); x,y,z=r['t']
    return [[R[0][0],R[0][1],R[0][2],x], [R[1][0],R[1][1],R[1][2],y], [R[2][0],R[2][1],R[2][2],z], [0,0,0,1]]
def mm(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
col_global={}
def comp_col(i):
    if i in col_global: return col_global[i]
    m=col_local_matrix(i)
    if parent[i] >= 0: m=mm(comp_col(parent[i]),m)
    col_global[i]=m; return m
for i in range(BONE_COUNT): comp_col(i)
global_pos={i:(col_global[i][0][3],col_global[i][1][3],col_global[i][2][3]) for i in range(BONE_COUNT)}
# Row-major FBX matrix from column global matrix: transpose rotation and put translation in final row.
def fbx_matrix_from_col(m):
    return [m[0][0],m[1][0],m[2][0],0, m[0][1],m[1][1],m[2][1],0, m[0][2],m[1][2],m[2][2],0, m[0][3],m[1][3],m[2][3],1]

def quat_to_euler_xyz_degrees(q):
    # Standard XYZ Euler approximation for FBX Lcl Rotation display.
    x,y,z,w=q
    # roll X
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch Y
    sinp = 2*(w*y - z*x)
    if abs(sinp) >= 1: pitch = math.copysign(math.pi/2, sinp)
    else: pitch = math.asin(sinp)
    # yaw Z
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

# Mesh skin bone IDs are stored as direct skeleton record indices in the vertex streams.
# No palette remapping is applied in this real-only pass.

# ------------------------- Mesh decode -------------------------
CMD_QUADS=0x80; CMD_TRIS=0x90; CMD_TRI_STRIP=0x98; CMD_TRI_FAN=0xA0
VALID={CMD_QUADS,CMD_TRIS,CMD_TRI_STRIP,CMD_TRI_FAN}
def read_display_list(start):
    pos=start; faces=[]; used=[]
    while pos < len(D)-3 and D[pos] in VALID:
        op=D[pos]; count=struct.unpack('>H',D[pos+1:pos+3])[0]; pos+=3
        verts=[]
        for _ in range(count):
            if pos+6 > len(D): break
            a,b,c=struct.unpack('>3H',D[pos:pos+6]); pos+=6
            # The three indices are position/normal/UV indices and are identical in this mesh stream.
            verts.append(a); used.append((a,b,c))
        if op==CMD_QUADS:
            for i in range(0,len(verts)-3,4): faces.append((verts[i],verts[i+1],verts[i+2])); faces.append((verts[i],verts[i+2],verts[i+3]))
        elif op==CMD_TRIS:
            for i in range(0,len(verts)-2,3): faces.append((verts[i],verts[i+1],verts[i+2]))
        elif op==CMD_TRI_STRIP:
            for i in range(len(verts)-2):
                a,b,c=verts[i],verts[i+1],verts[i+2]
                if a==b or b==c or a==c: continue
                faces.append((a,b,c) if i%2==0 else (b,a,c))
        elif op==CMD_TRI_FAN and len(verts)>=3:
            root=verts[0]
            for i in range(1,len(verts)-1):
                a,b,c=root,verts[i],verts[i+1]
                if a!=b and b!=c and a!=c: faces.append((a,b,c))
    return faces,used,pos

submeshes=[
    {'name':'Godzilla2K_detail_76byte', 'dl_start':0x0E2360, 'v_start':0x0E4FE0, 'v_stride':0x4C, 'v_count':971, 'layout':'blend76'},
    {'name':'Godzilla2K_body_64byte', 'dl_start':0x0F7040, 'v_start':0x1022E0, 'v_stride':0x40, 'v_count':3594, 'layout':'skin64'},
]

def map_slot(slot):
    slot=int(slot)
    if 0 <= slot < BONE_COUNT:
        return slot
    raise ValueError(f'Bone id {slot} outside decoded skeleton range 0..{BONE_COUNT-1}')
def norm3(v):
    l=math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) or 1.0
    return (v[0]/l,v[1]/l,v[2]/l)

def parse_skin64(off):
    x,y,z,w=struct.unpack('>4f',D[off:off+16])
    b0,b1=struct.unpack('>2H',D[off+16:off+20])
    u,v=struct.unpack('>2f',D[off+20:off+28])
    n=norm3(struct.unpack('>3f',D[off+28:off+40]))
    if b0==b1: pairs=[(map_slot(b0),1.0)]
    else: pairs=[(map_slot(b0),max(0,min(1,float(w)))), (map_slot(b1),max(0,min(1,1-float(w))))]
    s=sum(wt for _,wt in pairs) or 1.0
    pairs=[(b,wt/s) for b,wt in pairs if wt>1e-6]
    return (x,y,z),(u,1-v),n,pairs

def parse_blend76(off):
    x,y,z=struct.unpack('>3f',D[off:off+12])
    w0,w1,w2=struct.unpack('>3f',D[off+12:off+24])
    slots=struct.unpack('>4H',D[off+24:off+32])
    u,v=struct.unpack('>2f',D[off+32:off+40])
    n=norm3(struct.unpack('>3f',D[off+40:off+52]))
    # Exact 76-byte stream layout: three stored weights followed by four direct skeleton IDs.
    # The fourth influence is the residual weight. No guessed distribution is used.
    weights=[float(w0),float(w1),float(w2),float(1.0-(w0+w1+w2))]
    if any(w < -1e-5 or w > 1.0001 for w in weights):
        raise ValueError(f'Invalid 76-byte vertex weights at {off:#x}: {weights} slots={slots}')
    acc=collections.defaultdict(float)
    for slot,wt in zip(slots,weights):
        if abs(wt) <= 1e-6:
            continue
        acc[map_slot(slot)] += max(0.0, wt)
    total=sum(acc.values())
    if total <= 1e-8:
        raise ValueError(f'No positive 76-byte vertex weights at {off:#x}: {weights} slots={slots}')
    pairs=[(b,wt/total) for b,wt in sorted(acc.items())]
    return (x,y,z),(u,1-v),n,pairs

def get_vertex(sm, idx):
    if idx<0 or idx>=sm['v_count']: raise IndexError(idx)
    off=sm['v_start']+idx*sm['v_stride']
    return parse_skin64(off) if sm['layout']=='skin64' else parse_blend76(off)

vertices=[]; normals=[]; uvs=[]; vertex_weights=[]; poly_indices=[]; face_count=0; mesh_stats=[]
for sm in submeshes:
    faces, used, dl_end = read_display_list(sm['dl_start'])
    for face in faces:
        ids=[]
        for idx in face:
            pos,uv,nrm,wts=get_vertex(sm,idx)
            ids.append(len(vertices)); vertices.append(tuple(float(x) for x in pos)); uvs.append(tuple(float(x) for x in uv)); normals.append(tuple(float(x) for x in nrm)); vertex_weights.append(wts)
        poly_indices.extend([ids[0],ids[1],-ids[2]-1]); face_count+=1
    mesh_stats.append({'name':sm['name'],'display_list_start':hex(sm['dl_start']),'display_list_end':hex(dl_end),'vertex_start':hex(sm['v_start']),'vertex_stride':hex(sm['v_stride']),'vertex_count':sm['v_count'],'triangle_faces':len(faces),'used_index_min':min(x[0] for x in used),'used_index_max':max(x[0] for x in used),'all_index_triplets_identical':all(a==b==c for a,b,c in used)})

# ------------------------- Binary FBX writer -------------------------
class Prop:
    def __init__(self,code,value): self.code=code; self.value=value
class Arr:
    def __init__(self,code,values): self.code=code; self.values=values
class Node:
    def __init__(self,name,props=None,children=None): self.name=name; self.props=props or []; self.children=children or []
def PInt(v): return Prop('I',int(v))
def PLong(v): return Prop('L',int(v))
def PDouble(v): return Prop('D',float(v))
def PBool(v): return Prop('C',bool(v))
def PStr(v): return Prop('S',str(v))
def PRaw(v): return Prop('R',bytes(v))
def ADouble(vals): return Arr('d',list(vals))
def AInt(vals): return Arr('i',list(vals))
def ALong(vals): return Arr('l',list(vals))
def AFloat(vals): return Arr('f',list(vals))

def pack_prop(p):
    if isinstance(p,Prop):
        code=p.code.encode('ascii'); v=p.value
        if p.code=='I': return code+struct.pack('<i',v)
        if p.code=='L': return code+struct.pack('<q',v)
        if p.code=='D': return code+struct.pack('<d',v)
        if p.code=='C': return code+(b'\x01' if v else b'\x00')
        if p.code=='S':
            b=v.encode('utf-8'); return code+struct.pack('<I',len(b))+b
        if p.code=='R': return code+struct.pack('<I',len(v))+v
    if isinstance(p,Arr):
        vals=p.values; code=p.code.encode('ascii')
        if p.code=='d': data=struct.pack('<%sd'%len(vals),*map(float,vals)) if vals else b''
        elif p.code=='i': data=struct.pack('<%si'%len(vals),*map(int,vals)) if vals else b''
        elif p.code=='l': data=struct.pack('<%sq'%len(vals),*map(int,vals)) if vals else b''
        elif p.code=='f': data=struct.pack('<%sf'%len(vals),*map(float,vals)) if vals else b''
        else: raise ValueError(p.code)
        return code+struct.pack('<III',len(vals),0,len(data))+data
    raise TypeError(type(p))
NULL_RECORD=b'\x00'*13
def write_node(buf,node):
    start=buf.tell(); prop_bytes=b''.join(pack_prop(p) for p in node.props); name_bytes=node.name.encode('ascii')
    buf.write(b'\0'*12); buf.write(bytes([len(name_bytes)])); buf.write(name_bytes); buf.write(prop_bytes)
    for ch in node.children: write_node(buf,ch)
    if node.children: buf.write(NULL_RECORD)
    end=buf.tell(); cur=end; buf.seek(start); buf.write(struct.pack('<III',end,len(node.props),len(prop_bytes))); buf.write(bytes([len(name_bytes)])); buf.write(name_bytes); buf.seek(cur)

def p_node(name,ptype,label,flags,*values):
    props=[PStr(name),PStr(ptype),PStr(label),PStr(flags)]
    for v in values:
        if isinstance(v,bool): props.append(PBool(v))
        elif isinstance(v,int): props.append(PInt(v))
        elif isinstance(v,float): props.append(PDouble(v))
        else: props.append(PStr(v))
    return Node('P',props)
def flat3(vals):
    out=[]
    for a,b,c in vals: out += [a,b,c]
    return out
def flat2(vals):
    out=[]
    for a,b in vals: out += [a,b]
    return out
def identity(): return [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
def dist(a,b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

BASE_ID=2222222000; GEOM_ID=BASE_ID+1; MODEL_ID=BASE_ID+2; MAT_ID=BASE_ID+3; SKIN_ID=BASE_ID+4; POSE_ID=BASE_ID+5
TEX_ID_BASE=BASE_ID+100; VID_ID_BASE=BASE_ID+200; BONE_MODEL_BASE=BASE_ID+1000; BONE_ATTR_BASE=BASE_ID+2000; CLUSTER_BASE=BASE_ID+3000
geometry=Node('Geometry',[PLong(GEOM_ID),PStr('Geometry::Godzilla2K_Geometry'),PStr('Mesh')],[
    Node('Vertices',[ADouble(flat3(vertices))]),
    Node('PolygonVertexIndex',[AInt(poly_indices)]),
    Node('GeometryVersion',[PInt(124)]),
    Node('LayerElementNormal',[PInt(0)],[Node('Version',[PInt(101)]),Node('Name',[PStr('')]),Node('MappingInformationType',[PStr('ByPolygonVertex')]),Node('ReferenceInformationType',[PStr('Direct')]),Node('Normals',[ADouble(flat3(normals))])]),
    Node('LayerElementUV',[PInt(0)],[Node('Version',[PInt(101)]),Node('Name',[PStr('UVChannel_1')]),Node('MappingInformationType',[PStr('ByPolygonVertex')]),Node('ReferenceInformationType',[PStr('Direct')]),Node('UV',[ADouble(flat2(uvs))])]),
    Node('LayerElementMaterial',[PInt(0)],[Node('Version',[PInt(101)]),Node('Name',[PStr('')]),Node('MappingInformationType',[PStr('AllSame')]),Node('ReferenceInformationType',[PStr('IndexToDirect')]),Node('Materials',[AInt([0])])]),
    Node('Layer',[PInt(0)],[Node('Version',[PInt(100)]),Node('LayerElement',children=[Node('Type',[PStr('LayerElementNormal')]),Node('TypedIndex',[PInt(0)])]),Node('LayerElement',children=[Node('Type',[PStr('LayerElementUV')]),Node('TypedIndex',[PInt(0)])]),Node('LayerElement',children=[Node('Type',[PStr('LayerElementMaterial')]),Node('TypedIndex',[PInt(0)])])])
])
mesh_model=Node('Model',[PLong(MODEL_ID),PStr('Model::Godzilla2K'),PStr('Mesh')],[Node('Version',[PInt(232)]),Node('Properties70',children=[p_node('Lcl Translation','Lcl Translation','','A',0.0,0.0,0.0),p_node('Lcl Rotation','Lcl Rotation','','A',0.0,0.0,0.0),p_node('Lcl Scaling','Lcl Scaling','','A',1.0,1.0,1.0),p_node('DefaultAttributeIndex','int','Integer','',0)]),Node('Shading',[PBool(True)]),Node('Culling',[PStr('CullingOff')])])
material=Node('Material',[PLong(MAT_ID),PStr('Material::Godzilla2K_Material'),PStr('')],[Node('Version',[PInt(102)]),Node('ShadingModel',[PStr('phong')]),Node('MultiLayer',[PInt(0)]),Node('Properties70',children=[p_node('DiffuseColor','Color','','A',0.8,0.8,0.8),p_node('SpecularColor','Color','','A',0.25,0.25,0.25),p_node('BumpFactor','double','Number','A',0.45)])])
tex_bindings=[('Godzilla2k_G2_512_C','textures/Godzilla2k_G2_512_C.png','DiffuseColor'),('Godzilla2k_G2_512_N','textures/Godzilla2k_G2_512_N.png','NormalMap'),('Godzilla2k_G2_512_S','textures/Godzilla2k_G2_512_S.png','SpecularColor')]
texture_nodes=[]; video_nodes=[]
for i,(label,rel,prop) in enumerate(tex_bindings):
    tid=TEX_ID_BASE+i; vid=VID_ID_BASE+i; abs_file=str((OUT/rel).resolve())
    texture_nodes.append(Node('Texture',[PLong(tid),PStr(f'Texture::{label}'),PStr('')],[Node('Type',[PStr('TextureVideoClip')]),Node('Version',[PInt(202)]),Node('TextureName',[PStr(f'Texture::{label}')]),Node('Properties70',children=[p_node('WrapModeU','enum','','',0),p_node('WrapModeV','enum','','',0),p_node('UseMaterial','bool','','',1),p_node('UseMipMap','bool','','',1)]),Node('Media',[PStr(f'Video::{label}')]),Node('FileName',[PStr(abs_file)]),Node('RelativeFilename',[PStr(rel)]),Node('ModelUVTranslation',[PDouble(0),PDouble(0)]),Node('ModelUVScaling',[PDouble(1),PDouble(1)]),Node('Texture_Alpha_Source',[PStr('None')]),Node('Cropping',[PInt(0),PInt(0),PInt(0),PInt(0)])]))
    video_nodes.append(Node('Video',[PLong(vid),PStr(f'Video::{label}'),PStr('Clip')],[Node('Type',[PStr('Clip')]),Node('Properties70',children=[p_node('Path','KString','XRefUrl','',rel)]),Node('UseMipMap',[PInt(0)]),Node('Filename',[PStr(abs_file)]),Node('RelativeFilename',[PStr(rel)])]))

bone_models=[]; bone_attrs=[]
for i,name in enumerate(bone_names):
    r=skeleton[i]; tx,ty,tz=r['t']; rx,ry,rz=quat_to_euler_xyz_degrees(r['q'])
    # Limb length from first child, for display only.
    child_ids=[j for j,p in parent.items() if p==i]
    limb_len=3.0
    if child_ids: limb_len=max(0.5, dist(global_pos[child_ids[0]], global_pos[i]))
    bone_models.append(Node('Model',[PLong(BONE_MODEL_BASE+i),PStr(f'Model::{name}'),PStr('LimbNode')],[
        Node('Version',[PInt(232)]),
        Node('Properties70',children=[p_node('Lcl Translation','Lcl Translation','','A',float(tx),float(ty),float(tz)),p_node('Lcl Rotation','Lcl Rotation','','A',float(rx),float(ry),float(rz)),p_node('Lcl Scaling','Lcl Scaling','','A',1.0,1.0,1.0),p_node('RotationOrder','enum','','',0),p_node('LimbLength','double','Number','H',float(limb_len)),p_node('Size','double','Number','',1.0)]),
        Node('Shading',[PBool(True)]),Node('Culling',[PStr('CullingOff')])]))
    bone_attrs.append(Node('NodeAttribute',[PLong(BONE_ATTR_BASE+i),PStr(f'NodeAttribute::{name}'),PStr('LimbNode')],[Node('TypeFlags',[PStr('Skeleton')]),Node('Properties70',children=[p_node('Size','double','Number','',1.0)])]))

cluster_indices=collections.defaultdict(list); cluster_weights=collections.defaultdict(list)
for vidx,wts in enumerate(vertex_weights):
    for b,wt in wts:
        if 0 <= b < BONE_COUNT and wt > 1e-6:
            cluster_indices[b].append(vidx); cluster_weights[b].append(float(wt))
skin=Node('Deformer',[PLong(SKIN_ID),PStr('Deformer::Godzilla2K_Skin'),PStr('Skin')],[Node('Version',[PInt(101)]),Node('Link_DeformAcuracy',[PDouble(50.0)])])
clusters=[]
for i in range(BONE_COUNT):
    clusters.append(Node('Deformer',[PLong(CLUSTER_BASE+i),PStr(f'SubDeformer::Cluster_{bone_names[i]}'),PStr('Cluster')],[Node('Version',[PInt(100)]),Node('UserData',[PStr(''),PStr('')]),Node('Indexes',[AInt(cluster_indices.get(i,[]))]),Node('Weights',[ADouble(cluster_weights.get(i,[]))]),Node('Transform',[ADouble(identity())]),Node('TransformLink',[ADouble(fbx_matrix_from_col(col_global[i]))])]))
pose_children=[Node('Type',[PStr('BindPose')]),Node('Version',[PInt(100)]),Node('NbPoseNodes',[PInt(BONE_COUNT+1)]),Node('PoseNode',children=[Node('Node',[PLong(MODEL_ID)]),Node('Matrix',[ADouble(identity())])])]
for i in range(BONE_COUNT): pose_children.append(Node('PoseNode',children=[Node('Node',[PLong(BONE_MODEL_BASE+i)]),Node('Matrix',[ADouble(fbx_matrix_from_col(col_global[i]))])]))
pose=Node('Pose',[PLong(POSE_ID),PStr('Pose::Godzilla2K_BindPose'),PStr('BindPose')],pose_children)

# ------------------------- Animation decode (real delimited + continuation tracks) -------------------------
# Godzilla.BDG stores the named GODZILLA2K_* animation resources.  The first animation
# section is a rotation-track stream.  It contains two directly observed record layouts:
#
#   A) Explicit/headered track:
#      u16 header = (skeleton_bone_id << 8) | record_count, u16 zero
#      then record_count records: s16 qx, s16 qy, s16 qz, u16 normalized_time
#
#   B) Continuation/raw track between two explicit headers where exactly one skeleton
#      bone id is skipped:
#      record_count records: u16 normalized_time, s16 qx, s16 qy, s16 qz
#
# The previous animation package only exported layout A and dropped the loop-wrap key.
# That made many intermediate bones stay on their bind rotations and made cyclic clips
# start from the wrong pose.  This pass exports both proven layouts and sorts loop-wrap
# samples by their real normalized time instead of discarding them.
ANIM_D = ANIM_SRC.read_bytes()
ANIM_OBJ_BASE = 0x65e0
ANIM_RES_DESC = 0x5f40
ANIM_LOC_BASE = 0x76
ANIM_STR_BASE = 0x920
anim_str_count = struct.unpack('<I', ANIM_D[ANIM_STR_BASE:ANIM_STR_BASE+4])[0]
anim_ptrs = struct.unpack('<'+'I'*anim_str_count, ANIM_D[ANIM_STR_BASE+4:ANIM_STR_BASE+4+4*anim_str_count])
anim_strings=[]
for p in anim_ptrs:
    off=ANIM_STR_BASE+p; end=ANIM_D.find(b'\0',off)
    anim_strings.append(ANIM_D[off:end].decode('latin1', errors='replace'))

def anim_res_name(rid):
    name_idx=struct.unpack('>I', ANIM_D[ANIM_RES_DESC+rid*16+4:ANIM_RES_DESC+rid*16+8])[0]
    return anim_strings[name_idx]
def anim_res_loc(rid):
    rid2,rel,size=struct.unpack('>III', ANIM_D[ANIM_LOC_BASE+rid*18:ANIM_LOC_BASE+rid*18+12])
    return ANIM_OBJ_BASE+rel,size

def quat_from_stored_xyz(qx_i, qy_i, qz_i):
    qx=qx_i/32767.0; qy=qy_i/32767.0; qz=qz_i/32767.0
    n2=qx*qx+qy*qy+qz*qz
    if n2 > 1.05:
        return None
    qw=math.sqrt(max(0.0, 1.0-n2))
    ln=math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw) or 1.0
    return (qx/ln, qy/ln, qz/ln, qw/ln)

def validate_track_times(raw_times):
    if len(raw_times) < 2 or len(set(raw_times)) < 2:
        return False
    if any(t < 0 or t > 65535 for t in raw_times):
        return False
    drops=[i for i,(a,b) in enumerate(zip(raw_times, raw_times[1:])) if b+16 < a]
    # Either naturally monotonic or one final loop-wrap record.
    if len(drops) > 1:
        return False
    if len(drops) == 1:
        if drops[0] != len(raw_times)-2:
            return False
        if raw_times[-2] < 55000 or raw_times[-1] > 30000:
            return False
        if any(b <= a for a,b in zip(raw_times[:-2], raw_times[1:-1])):
            return False
    else:
        if any(b <= a for a,b in zip(raw_times, raw_times[1:])):
            return False
    return True

def sort_and_scale_keys(raw_keys, duration):
    # Keep the real wrapped sample, but place it by its stored normalized timestamp.
    # Duplicate timestamps are rejected rather than merged/invented.
    ordered=sorted(raw_keys, key=lambda x: x[0])
    if any(b[0] <= a[0] for a,b in zip(ordered, ordered[1:])):
        return None
    dur=float(duration if duration > 0 else 1.0)
    return [(float(t)/65534.0*dur, q, raw_tuple) for (t,q,raw_tuple) in ordered]

def decode_explicit_track_at(base_off, rel, h38, duration):
    if rel+4 > h38:
        return None
    hw, zero = struct.unpack('>HH', ANIM_D[base_off+rel:base_off+rel+4])
    bone = hw >> 8
    count = hw & 0xff
    if zero != 0 or bone < 0 or bone >= BONE_COUNT or count < 3 or count > 160:
        return None
    if rel + 4 + count*8 > h38:
        return None
    raw_times=[]; raw_keys=[]; max_q_norm2=0.0
    for k in range(count):
        qx_i,qy_i,qz_i,t_u = struct.unpack('>hhhH', ANIM_D[base_off+rel+4+k*8:base_off+rel+4+k*8+8])
        q=quat_from_stored_xyz(qx_i,qy_i,qz_i)
        if q is None:
            return None
        qx=qx_i/32767.0; qy=qy_i/32767.0; qz=qz_i/32767.0
        max_q_norm2=max(max_q_norm2, qx*qx+qy*qy+qz*qz)
        raw_times.append(int(t_u)); raw_keys.append((int(t_u), q, (int(qx_i),int(qy_i),int(qz_i),int(t_u))))
    if not validate_track_times(raw_times):
        return None
    keys=sort_and_scale_keys(raw_keys, duration)
    if not keys or len(keys) < 2:
        return None
    return {'bone':bone,'rel':rel,'end':rel+4+count*8,'count':count,'keys':keys,'raw_times':raw_times,'layout':'explicit_qxyz_time','max_q_norm2':max_q_norm2}

def decode_continuation_track(base_off, start_rel, end_rel, bone, duration):
    if start_rel >= end_rel or (end_rel-start_rel) % 8 != 0:
        return None
    count=(end_rel-start_rel)//8
    if bone < 0 or bone >= BONE_COUNT or count < 3 or count > 160:
        return None
    raw_times=[]; raw_keys=[]; max_q_norm2=0.0
    for rel in range(start_rel, end_rel, 8):
        t_u,qx_i,qy_i,qz_i = struct.unpack('>Hhhh', ANIM_D[base_off+rel:base_off+rel+8])
        q=quat_from_stored_xyz(qx_i,qy_i,qz_i)
        if q is None:
            return None
        qx=qx_i/32767.0; qy=qy_i/32767.0; qz=qz_i/32767.0
        max_q_norm2=max(max_q_norm2, qx*qx+qy*qy+qz*qz)
        raw_times.append(int(t_u)); raw_keys.append((int(t_u), q, (int(t_u),int(qx_i),int(qy_i),int(qz_i))))
    if not validate_track_times(raw_times):
        return None
    keys=sort_and_scale_keys(raw_keys, duration)
    if not keys or len(keys) < 2:
        return None
    return {'bone':bone,'rel':start_rel,'end':end_rel,'count':count,'keys':keys,'raw_times':raw_times,'layout':'continuation_time_qxyz','max_q_norm2':max_q_norm2}


def find_terminal_bone_id_table(base_off, rot_end):
    """Find the terminal 16-bit bone-id list often stored at the end of the primary
    rotation section (e.g. 34 00 35 00 ... 44 00). This is not animation data;
    it marks where a final time-first continuation track must stop.
    """
    search_start=max(0x58, rot_end-0x400)
    best=None
    for rel in range(search_start, rot_end-8, 2):
        ids=[]; p=rel
        while p+2 <= rot_end:
            b=ANIM_D[base_off+p]
            z=ANIM_D[base_off+p+1]
            if z != 0 or b < 0 or b >= BONE_COUNT:
                break
            if ids and b == 0:
                break
            ids.append(b); p += 2
        if len(ids) >= 4 and all(ids[i]+1 == ids[i+1] for i in range(len(ids)-1)):
            # Prefer the earliest high-quality run close to the end.
            best=rel
            break
    return best

def decode_primary_animation_tracks(rid):
    off,size=anim_res_loc(rid)
    name=anim_res_name(rid)
    duration=struct.unpack('>f', ANIM_D[off+0x1c:off+0x20])[0]
    rot_end=struct.unpack('>I', ANIM_D[off+0x38:off+0x3c])[0]
    table_start=find_terminal_bone_id_table(off, rot_end)
    scan_end=table_start if table_start is not None else rot_end
    candidates=[]
    for rel in range(0x58, max(0x58, scan_end-4), 4):
        tr=decode_explicit_track_at(off, rel, scan_end, duration)
        if tr:
            candidates.append(tr)
    explicit=[]; last_end=0
    for tr in candidates:
        # Reject obvious false positives that jump backward in skeleton order after
        # established higher bone tracks. This keeps bogus tiny-count matches inside
        # long streams out of Blender without inventing any motion.
        if tr['rel'] >= last_end:
            if explicit and tr['bone'] < explicit[-1]['bone']:
                continue
            explicit.append(tr)
            last_end=tr['end']
    tracks=[]
    terminal_continuation=None
    for i,tr in enumerate(explicit):
        tracks.append(tr)
        if i+1 >= len(explicit):
            continue
        next_tr=explicit[i+1]
        gap_start=tr['end']; gap_end=next_tr['rel']
        # The only non-ambiguous continuation case: exactly one bone id is skipped
        # between two explicit headers, and the byte gap validates as time-first
        # quaternion records for that skipped bone.
        if next_tr['bone'] - tr['bone'] == 2:
            cont=decode_continuation_track(off, gap_start, gap_end, tr['bone']+1, duration)
            if cont:
                tracks.append(cont)
    # Some clips end with an explicit leg/twist track, then a final time-first
    # continuation track immediately before the terminal bone-id table.  The v2
    # exporter stopped at the last explicit header and missed that real track
    # (commonly the final femur twist / transition bone). Decode it only when the
    # whole remaining byte gap validates cleanly.
    if explicit:
        gap_start=explicit[-1]['end']
        gap_end=table_start if table_start is not None else scan_end
        if gap_end and gap_end > gap_start and (gap_end-gap_start) % 8 == 0:
            cont=decode_continuation_track(off, gap_start, gap_end, explicit[-1]['bone']+1, duration)
            if cont:
                terminal_continuation=cont
                tracks.append(cont)
    tracks=sorted(tracks, key=lambda x: (x['rel'], x['bone']))
    # One rotation track per skeleton bone per clip.  A second track for the same
    # bone in this byte stream is treated as an accidental header match inside
    # channel/table data, so it is not exported.
    unique=[]; seen_bones=set()
    for tr in tracks:
        if tr['bone'] in seen_bones:
            continue
        seen_bones.add(tr['bone'])
        unique.append(tr)
    return {'rid':rid,'name':name,'duration':duration,'size':size,'rot_section_end':rot_end,'terminal_bone_table_start':table_start,'explicit_tracks':explicit,'terminal_continuation':terminal_continuation,'tracks':unique}

anim_decoded=[]
for rid in range(3,104):
    if anim_res_name(rid).startswith('GODZILLA2K_'):
        a=decode_primary_animation_tracks(rid)
        if a['tracks']:
            anim_decoded.append(a)

FBX_TICKS_PER_SECOND = 46186158000
ANIM_ID_BASE = BASE_ID + 1000000
animation_objects=[]
animation_connections=[]
def p_time(name, value):
    return Node('P',[PStr(name),PStr('KTime'),PStr('Time'),PStr(''),PLong(int(value))])
anim_curve_count=0; anim_curve_node_count=0; anim_stack_count=0; anim_layer_count=0
anim_manifest=[]

def make_anim_curve(curve_id, name, times_sec, values):
    global anim_curve_count
    anim_curve_count += 1
    key_times=[int(round(t*FBX_TICKS_PER_SECOND)) for t in times_sec]
    vals=[float(v) for v in values]
    n=len(vals)
    # Blender's FBX importer expects one shared key-attribute block by default, not
    # one KeyAttrFlags/KeyAttrDataFloat tuple per key.  The earlier export used
    # per-key attribute arrays and Blender rejected the file even though the
    # binary node structure itself was valid.
    return Node('AnimationCurve',[PLong(curve_id),PStr(f'AnimCurve::{name}'),PStr('')],[
        Node('Default',[PDouble(0.0)]),
        Node('KeyVer',[PInt(4008)]),
        Node('KeyTime',[ALong(key_times)]),
        Node('KeyValueFloat',[AFloat(vals)]),
        Node('KeyAttrFlags',[AInt([24840])]),
        Node('KeyAttrDataFloat',[AFloat([0.0,0.0,0.0,0.0])]),
        Node('KeyAttrRefCount',[AInt([n])])
    ])

# Export every strictly validated direct quaternion rotation track found in each clip.
for ai,a in enumerate(anim_decoded):
    stack_id=ANIM_ID_BASE + ai*10000 + 1
    layer_id=ANIM_ID_BASE + ai*10000 + 2
    stack=Node('AnimationStack',[PLong(stack_id),PStr(f'AnimStack::{a["name"]}'),PStr('')],[Node('Properties70',children=[p_time('LocalStart',0),p_time('LocalStop',int(round(a['duration']*FBX_TICKS_PER_SECOND))),p_time('ReferenceStart',0),p_time('ReferenceStop',int(round(a['duration']*FBX_TICKS_PER_SECOND)))])])
    layer=Node('AnimationLayer',[PLong(layer_id),PStr(f'AnimLayer::{a["name"]}_Layer'),PStr('')])
    animation_objects += [stack, layer]
    anim_stack_count += 1; anim_layer_count += 1
    animation_connections.append(Node('C',[PStr('OO'),PLong(layer_id),PLong(stack_id)]))
    man_tracks=[]
    for ti,tr in enumerate(a['tracks']):
        bone=tr['bone']; bname=bone_names[bone]
        times=[k[0] for k in tr['keys']]
        quats=[k[1] for k in tr['keys']]
        # If a real track starts after t=0, keep the game skeleton's real bind/rest
        # rotation at frame 0. This prevents Blender from evaluating the animated
        # property from an implicit zero rotation before the first stored key.
        if not times or times[0] > 1e-7:
            times=[0.0]+times
            quats=[skeleton[bone]['q']]+quats
        eulers=[quat_to_euler_xyz_degrees(q) for q in quats]
        # Three FBX rotation curves for this bone.
        cnode_id=ANIM_ID_BASE + ai*10000 + 100 + ti
        cnode=Node('AnimationCurveNode',[PLong(cnode_id),PStr(f'AnimCurveNode::{a["name"]}_{bname}_R'),PStr('')],[Node('Properties70',children=[p_node('d|X','Number','','A',0.0),p_node('d|Y','Number','','A',0.0),p_node('d|Z','Number','','A',0.0)])])
        animation_objects.append(cnode); anim_curve_node_count += 1
        animation_connections.append(Node('C',[PStr('OO'),PLong(cnode_id),PLong(layer_id)]))
        animation_connections.append(Node('C',[PStr('OP'),PLong(cnode_id),PLong(BONE_MODEL_BASE+bone),PStr('Lcl Rotation')]))
        for axis,idx in [('X',0),('Y',1),('Z',2)]:
            cid=ANIM_ID_BASE + ai*10000 + 1000 + ti*10 + idx
            curve=make_anim_curve(cid, f'{a["name"]}_{bname}_R_{axis}', times, [e[idx] for e in eulers])
            animation_objects.append(curve)
            animation_connections.append(Node('C',[PStr('OP'),PLong(cid),PLong(cnode_id),PStr(f'd|{axis}')]))
        man_tracks.append({'bone_id':bone,'bone_name':bname,'track_rel':hex(tr['rel']),'layout':tr['layout'],'record_count':tr['count'],'exported_key_count':len(tr['keys']),'raw_times_start':tr['raw_times'][:5],'raw_times_end':tr['raw_times'][-5:],'loop_wrap_sample_sorted_not_omitted':True,'bind_rest_key_inserted_at_t0_if_needed':True})
    anim_manifest.append({'resource_id':a['rid'],'name':a['name'],'duration_seconds':a['duration'],'resource_size':hex(a['size']),'rotation_section_end':hex(a['rot_section_end']),'terminal_bone_table_start':(hex(a['terminal_bone_table_start']) if a.get('terminal_bone_table_start') is not None else None),'terminal_continuation_added':bool(a.get('terminal_continuation')),'exported_tracks':man_tracks})

objects=Node('Objects',children=[geometry,mesh_model,material]+texture_nodes+video_nodes+bone_models+bone_attrs+[skin]+clusters+[pose]+animation_objects)
connections=[]
connections.append(Node('C',[PStr('OO'),PLong(MODEL_ID),PLong(0)])); connections.append(Node('C',[PStr('OO'),PLong(GEOM_ID),PLong(MODEL_ID)])); connections.append(Node('C',[PStr('OO'),PLong(MAT_ID),PLong(MODEL_ID)]))
for i,(label,rel,prop) in enumerate(tex_bindings): connections.append(Node('C',[PStr('OP'),PLong(TEX_ID_BASE+i),PLong(MAT_ID),PStr(prop)])); connections.append(Node('C',[PStr('OO'),PLong(VID_ID_BASE+i),PLong(TEX_ID_BASE+i)]))
for i in range(BONE_COUNT):
    connections.append(Node('C',[PStr('OO'),PLong(BONE_ATTR_BASE+i),PLong(BONE_MODEL_BASE+i)]))
    pid=parent[i]
    connections.append(Node('C',[PStr('OO'),PLong(BONE_MODEL_BASE+i),PLong(BONE_MODEL_BASE+pid if pid>=0 else 0)]))
connections.append(Node('C',[PStr('OO'),PLong(SKIN_ID),PLong(GEOM_ID)]))
for i in range(BONE_COUNT): connections.append(Node('C',[PStr('OO'),PLong(CLUSTER_BASE+i),PLong(SKIN_ID)])); connections.append(Node('C',[PStr('OO'),PLong(BONE_MODEL_BASE+i),PLong(CLUSTER_BASE+i)]))
connections.extend(animation_connections)
connections=Node('Connections',children=connections)
def object_type(n,c): return Node('ObjectType',[PStr(n)],[Node('Count',[PInt(c)])])
definitions=Node('Definitions',children=[Node('Version',[PInt(100)]),Node('Count',[PInt(3+len(texture_nodes)+len(video_nodes)+BONE_COUNT*3+2+len(animation_objects))]),object_type('Model',1+BONE_COUNT),object_type('Geometry',1),object_type('Material',1),object_type('Texture',len(texture_nodes)),object_type('Video',len(video_nodes)),object_type('NodeAttribute',BONE_COUNT),object_type('Deformer',1+BONE_COUNT),object_type('Pose',1),object_type('AnimationStack',anim_stack_count),object_type('AnimationLayer',anim_layer_count),object_type('AnimationCurveNode',anim_curve_node_count),object_type('AnimationCurve',anim_curve_count)])
global_settings=Node('GlobalSettings',children=[Node('Version',[PInt(1000)]),Node('Properties70',children=[p_node('UpAxis','int','Integer','',2),p_node('UpAxisSign','int','Integer','',1),p_node('FrontAxis','int','Integer','',1),p_node('FrontAxisSign','int','Integer','',-1),p_node('CoordAxis','int','Integer','',0),p_node('CoordAxisSign','int','Integer','',1),p_node('UnitScaleFactor','double','Number','',1.0),p_node('OriginalUnitScaleFactor','double','Number','',1.0)])])
header_ext=Node('FBXHeaderExtension',children=[Node('FBXHeaderVersion',[PInt(1003)]),Node('FBXVersion',[PInt(7400)]),Node('EncryptionType',[PInt(0)]),Node('Creator',[PStr('Godzilla2K real skeleton/mesh plus terminal continuation + boundary keys v3')])])
takes_children=[Node('Current',[PStr('')])]
for a in anim_decoded:
    ticks=int(round(a['duration']*FBX_TICKS_PER_SECOND))
    takes_children.append(Node('Take',[PStr(a['name'])],[Node('FileName',[PStr(f'{a["name"]}.tak')]),Node('LocalTime',[PLong(0),PLong(ticks)]),Node('ReferenceTime',[PLong(0),PLong(ticks)])]))
takes_node=Node('Takes',children=takes_children)
root_nodes=[header_ext,Node('FileId',[PRaw(b'\0'*16)]),global_settings,definitions,objects,connections,takes_node]
fbx=io.BytesIO(); fbx.write(b'Kaydara FBX Binary  \x00\x1a\x00'); fbx.write(struct.pack('<I',7400))
for n in root_nodes: write_node(fbx,n)
fbx.write(NULL_RECORD); fbx.write(b'\0'*160)
(OUT/'Godzilla2K.fbx').write_bytes(fbx.getvalue())
manifest={
    'profile':'Godzilla2K_proven_offsets_v3',
    'tool_note':'Export is real-data-only for the proven Godzilla2K profile. Other kaiju need profile discovery/generalization.',
    'source':str(SRC.name), 'animation_source':str(ANIM_SRC.name), 'pvms':[p.name for p in PVM_FILES], 'fbx':'Godzilla2K.fbx', 'triangles':face_count, 'control_points':len(vertices),
    'skeleton':'Decoded from real GODZILLA2K_SKELETON block at 0x0EE0/0x0F20',
    'bone_count':BONE_COUNT,
    'weighted_bones':len([i for i in range(BONE_COUNT) if cluster_indices.get(i)]),
    'mesh_stats':mesh_stats,
    'uvs':'Decoded from vertex streams; V flipped for Blender/FBX image orientation',
    'skin64_weights':'Exact two-influence skin weights from 64-byte vertex stream: stored weight plus residual second weight; direct skeleton IDs; no remap',
    'blend76_weights':'Exact four-influence skin weights from 76-byte vertex stream: three stored floats plus residual fourth weight; direct skeleton IDs; no remap/no distribution guessing',
    'bones':[{'idx':i,'name':bone_names[i],'parent':parent[i],'local_translation':skeleton[i]['t'],'local_quaternion_xyzw':skeleton[i]['q'],'global_position':global_pos[i],'influenced_vertices':len(cluster_indices.get(i,[]))} for i in range(BONE_COUNT)],
    'animations': anim_manifest,
    'animation_export_scope':'FBX contains strictly validated explicit quaternion rotation tracks plus non-ambiguous continuation rotation tracks, including terminal continuation tracks before terminal bone-id tables when cleanly validated. Rest/bind keys are inserted at t=0 only when the game track has no time-zero key. Translation/root-motion and unproven long-count/compressed sections are preserved in animations_raw but not attached.',
    'shape_texture_data_base': hex(DATA_BASE),
    'texture_specs': [
        {'filename': name, 'format': fmt, 'relative_offset': hex(rel), 'absolute_offset': hex(DATA_BASE+rel), 'width': w, 'height': h, 'source_file': SRC.name}
        for name,(fmt,rel,w,h) in texture_specs.items()
    ],
    'animation_resource_locations': [
        {'resource_id': a['rid'], 'name': a['name'], 'source_file': ANIM_SRC.name, 'absolute_offset': hex(anim_res_loc(a['rid'])[0]), 'size': anim_res_loc(a['rid'])[1]}
        for a in anim_decoded
    ],
}
# Preserve raw animation resources and decoded metadata for the next importer/exporter pass.
RAW_ANIM_DIR = OUT/'animations_raw'
RAW_ANIM_DIR.mkdir(exist_ok=True)
for a in anim_decoded:
    off,size=anim_res_loc(a['rid'])
    (RAW_ANIM_DIR/f"{a['name']}.bin").write_bytes(ANIM_D[off:off+size])
(RAW_ANIM_DIR/'animation_tracks_manifest.json').write_text(json.dumps(anim_manifest, indent=2))

# Hash exported files so Import.bat can patch only files the user changed.
def _sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
manifest['file_hashes'] = {'Godzilla2K.fbx': _sha256_file(OUT/'Godzilla2K.fbx')}
for tex in sorted(TEX.iterdir()):
    if tex.suffix.lower()=='.png':
        manifest['file_hashes'][f'textures/{tex.name}'] = _sha256_file(tex)
for raw in sorted(RAW_ANIM_DIR.iterdir()):
    if raw.is_file():
        manifest['file_hashes'][f'animations_raw/{raw.name}'] = _sha256_file(raw)
manifest['importer_contract'] = {
    'same_topology_fbx_required': True,
    'mesh_writeback': 'positions, normals, UVs, and weights are imported from FBX only when polygon-vertex count/order still matches the extracted display lists',
    'skeleton_writeback': 'bone Lcl Translation and Lcl Rotation are imported from FBX LimbNode models by bone name',
    'texture_writeback': 'changed PNGs are encoded to Wii CMPR, RGB565, or I8 and written to Shapes.BDG at manifest offsets',
    'animation_writeback': 'animations_raw/*.bin can be same-size swapped; Blender animation curves are not yet encoded back to BDG animation channels'
}

(OUT/'decode_manifest.json').write_text(json.dumps(manifest,indent=2))
zip_path=OUT.with_suffix('.zip')
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    z.write(OUT/'Godzilla2K.fbx','Godzilla2K.fbx')
    z.write(OUT/'decode_manifest.json','decode_manifest.json')
    for tex in sorted(TEX.iterdir()):
        if tex.suffix.lower()=='.png': z.write(tex,f'textures/{tex.name}')
    raw_dir=OUT/'animations_raw'
    if raw_dir.exists():
        for raw in sorted(raw_dir.iterdir()):
            z.write(raw, f'animations_raw/{raw.name}')
print('wrote',zip_path)
print(json.dumps({'triangles':face_count,'control_points':len(vertices),'bones':BONE_COUNT,'weighted_bones':manifest['weighted_bones'],'mesh_stats':mesh_stats},indent=2))
