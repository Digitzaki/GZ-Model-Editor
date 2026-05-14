#!/usr/bin/env python3
"""
BDG/PVM reimporter for the current proven Godzilla2K BDG-to-FBX bridge.

This version performs real binary writeback for the pieces that are understood:
  * copies the original BDG/PVM files into <name>-Kaiju-Reimported/
  * re-encodes changed PNG textures back to the Shapes.BDG texture data area
    (CMPR/DXT1, RGB565, and I8, in Wii tiled order)
  * imports same-topology FBX mesh edits back into the original vertex streams
    (positions, normals, UVs, and skin weights where clusters are available)
  * imports FBX rest-pose bone translation/rotation back into the skeleton records
  * replaces animations_raw/*.bin only when the byte size is exactly unchanged

It still does not synthesize proprietary BDG animation streams from Blender action
curves. That requires a proven encoder for the primary/secondary animation layouts.
"""
from __future__ import annotations
from pathlib import Path
import argparse, collections, hashlib, json, math, os, shutil, struct, sys, zlib
from PIL import Image

# ----------------------------- small utilities -----------------------------

def clean_windows_folder_arg(value: str) -> str:
    value = str(value).strip()
    while value and value[-1] in ('\"', "'"):
        value = value[:-1].rstrip()
    while value and value[0] in ('\"', "'"):
        value = value[1:].lstrip()
    return value or '.'

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def find_case_insensitive(path: Path, name: str) -> Path | None:
    p = path / name
    if p.exists(): return p
    lname = name.lower()
    for q in path.iterdir():
        if q.name.lower() == lname: return q
    return None

def parse_int_maybe_hex(v) -> int:
    if isinstance(v, int): return v
    return int(str(v), 16 if str(v).lower().startswith('0x') else 10)

def find_extract_folder(root: Path) -> Path:
    candidates = sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith('-Kaiju-Extracted')])
    if not candidates:
        raise SystemExit('No *-Kaiju-Extracted folder found beside Import.bat.')
    if len(candidates) > 1:
        raise SystemExit('More than one extracted folder found. Keep one kaiju project beside Import.bat at a time:\n  ' + '\n  '.join(p.name for p in candidates))
    return candidates[0]

def load_manifest(extracted: Path) -> dict:
    m = extracted / 'decode_manifest.json'
    if not m.exists():
        raise SystemExit(f'Missing decode_manifest.json in {extracted}')
    return json.loads(m.read_text(encoding='utf-8'))

# ----------------------------- Wii texture encoders -----------------------------

def rgb565_pack(r:int,g:int,b:int) -> int:
    r5=max(0,min(31,round(r*31/255)))
    g6=max(0,min(63,round(g*63/255)))
    b5=max(0,min(31,round(b*31/255)))
    return (r5<<11)|(g6<<5)|b5

def rgb565_unpack(c:int):
    return (((c>>11)&31)*255//31, ((c>>5)&63)*255//63, (c&31)*255//31, 255)

def encode_rgb565_png(path: Path, width: int, height: int) -> bytes:
    img = Image.open(path).convert('RGB').resize((width, height))
    pix = img.load(); out=bytearray()
    # Wii RGB565 texture tiles are 4x4 pixels.
    for ty in range(0,height,4):
      for tx in range(0,width,4):
        for y in range(4):
          for x in range(4):
            xx,yy=tx+x,ty+y
            r,g,b = pix[xx,yy] if xx<width and yy<height else (0,0,0)
            out += struct.pack('>H', rgb565_pack(r,g,b))
    return bytes(out)

def encode_i8_png(path: Path, width: int, height: int) -> bytes:
    img = Image.open(path).convert('L').resize((width, height))
    pix = img.load(); out=bytearray()
    # Wii I8 texture tiles are 8x4 pixels.
    for ty in range(0,height,4):
      for tx in range(0,width,8):
        for y in range(4):
          for x in range(8):
            xx,yy=tx+x,ty+y
            out.append(pix[xx,yy] if xx<width and yy<height else 0)
    return bytes(out)

def nearest_palette_index(rgb, pal, allow_alpha=False, alpha=255):
    if allow_alpha and alpha < 128:
        return 3
    r,g,b=rgb
    best_i=0; best_d=10**18
    max_i=3 if len(pal)>=4 else len(pal)-1
    for i in range(max_i+1):
        pr,pg,pb,pa=pal[i]
        if allow_alpha and i==3 and pa==0:
            continue
        d=(r-pr)*(r-pr)+(g-pg)*(g-pg)+(b-pb)*(b-pb)
        if d<best_d:
            best_d=d; best_i=i
    return best_i

def dxt1_block_encode(pixels_rgba):
    opaque = all(a >= 128 for _,_,_,a in pixels_rgba)
    colors=[(r,g,b) for r,g,b,a in pixels_rgba if opaque or a>=128]
    if not colors: colors=[(0,0,0)]
    # Simple deterministic endpoint selection by luminance. This is a valid CMPR encoder,
    # not a quality-optimized compressor.
    def lum(c): return c[0]*0.299+c[1]*0.587+c[2]*0.114
    cmin=min(colors, key=lum); cmax=max(colors, key=lum)
    q0=rgb565_pack(*cmax); q1=rgb565_pack(*cmin)
    if opaque and q0 == q1:
        # keep opaque 4-color mode by making c0 > c1 when possible
        q0 = min(0xFFFF, q0+1) if q0 <= q1 else q0
    if opaque:
        if q0 <= q1: q0,q1=q1,q0
        p0=rgb565_unpack(q0); p1=rgb565_unpack(q1)
        pal=[p0,p1,
             tuple((2*p0[i]+p1[i])//3 for i in range(3))+(255,),
             tuple((p0[i]+2*p1[i])//3 for i in range(3))+(255,)]
        allow_alpha=False
    else:
        if q0 > q1: q0,q1=q1,q0
        p0=rgb565_unpack(q0); p1=rgb565_unpack(q1)
        pal=[p0,p1,tuple((p0[i]+p1[i])//2 for i in range(3))+(255,),(0,0,0,0)]
        allow_alpha=True
    bits=0
    for i,(r,g,b,a) in enumerate(pixels_rgba):
        idx=nearest_palette_index((r,g,b), pal, allow_alpha, a)
        bits |= (idx & 3) << (30 - 2*i)
    return struct.pack('>HHI', q0, q1, bits)

def encode_cmpr_png(path: Path, width: int, height: int) -> bytes:
    img = Image.open(path).convert('RGBA').resize((width, height))
    pix=img.load(); out=bytearray()
    # Wii CMPR macroblocks are 8x8 pixels holding four DXT1 blocks: TL, TR, BL, BR.
    for y0 in range(0,height,8):
      for x0 in range(0,width,8):
        for by,bx in [(0,0),(0,4),(4,0),(4,4)]:
          block=[]
          for y in range(4):
            for x in range(4):
              xx=x0+bx+x; yy=y0+by+y
              block.append(pix[xx,yy] if xx<width and yy<height else (0,0,0,0))
          out += dxt1_block_encode(block)
    return bytes(out)

# ----------------------------- FBX binary parser -----------------------------

class FbxNode:
    __slots__=('name','props','children')
    def __init__(self,name,props=None,children=None):
        self.name=name; self.props=props or []; self.children=children or []
    def child(self, name):
        for c in self.children:
            if c.name == name: return c
        return None
    def children_named(self, name):
        return [c for c in self.children if c.name == name]

def _read_prop(data: bytes, pos: int):
    code=chr(data[pos]); pos += 1
    if code == 'Y': return struct.unpack_from('<h', data, pos)[0], pos+2
    if code == 'C': return bool(data[pos]), pos+1
    if code == 'I': return struct.unpack_from('<i', data, pos)[0], pos+4
    if code == 'F': return struct.unpack_from('<f', data, pos)[0], pos+4
    if code == 'D': return struct.unpack_from('<d', data, pos)[0], pos+8
    if code == 'L': return struct.unpack_from('<q', data, pos)[0], pos+8
    if code in ('S','R'):
        n=struct.unpack_from('<I', data, pos)[0]; pos+=4
        raw=data[pos:pos+n]; pos+=n
        if code == 'S':
            return raw.decode('utf-8', errors='replace'), pos
        return raw, pos
    if code in ('f','d','i','l','b','c'):
        count, encoding, byte_len = struct.unpack_from('<III', data, pos); pos += 12
        raw=data[pos:pos+byte_len]; pos += byte_len
        if encoding == 1: raw = zlib.decompress(raw)
        if code == 'f': fmt='<%df'%count
        elif code == 'd': fmt='<%dd'%count
        elif code == 'i': fmt='<%di'%count
        elif code == 'l': fmt='<%dq'%count
        elif code in ('b','c'): fmt='<%d?'%count
        else: raise ValueError(code)
        return list(struct.unpack(fmt, raw)) if count else [], pos
    raise ValueError(f'Unsupported FBX property code {code!r} at {pos-1:#x}')

def parse_fbx(path: Path):
    data=path.read_bytes()
    if not data.startswith(b'Kaydara FBX Binary'):
        raise ValueError('Only binary FBX is supported by this importer.')
    version=struct.unpack_from('<I', data, 23)[0]
    pos=27; use64=version >= 7500
    def read_node(pos:int):
        if use64:
            if pos+25 > len(data): return None,pos
            end,num_props,prop_len=struct.unpack_from('<QQQ', data, pos); pos += 24
            name_len=data[pos]; pos += 1
            null_size=25
        else:
            if pos+13 > len(data): return None,pos
            end,num_props,prop_len=struct.unpack_from('<III', data, pos); pos += 12
            name_len=data[pos]; pos += 1
            null_size=13
        if end == 0 and num_props == 0 and prop_len == 0 and name_len == 0:
            return None, pos
        name=data[pos:pos+name_len].decode('ascii', errors='replace'); pos += name_len
        props=[]
        for _ in range(num_props):
            val,pos=_read_prop(data,pos); props.append(val)
        children=[]
        while pos < end - null_size:
            child,pos2=read_node(pos)
            if child is None:
                pos=pos2; break
            children.append(child); pos=pos2
        pos=end
        return FbxNode(name,props,children),pos
    roots=[]
    while pos < len(data):
        n,pos2=read_node(pos)
        if n is None: break
        roots.append(n); pos=pos2
    return roots,version

def walk_nodes(nodes):
    for n in nodes:
        yield n
        yield from walk_nodes(n.children)

def find_first(nodes, name):
    for n in walk_nodes(nodes):
        if n.name == name: return n
    return None

def object_nodes(nodes, typename=None):
    objects=find_first(nodes,'Objects')
    if not objects: return []
    if typename is None: return objects.children
    return [n for n in objects.children if n.name == typename]

def clean_fbx_object_name(s: str) -> str:
    # FBX names often look like "Model::Bip01" or may contain namespace/model separators.
    s=str(s)
    if '::' in s: s=s.split('::',1)[1]
    s=s.replace('\x00\x01','')
    if '|' in s: s=s.split('|')[-1]
    if ':' in s: s=s.split(':')[-1]
    return s

def p_values(properties70: FbxNode, prop_name: str):
    if not properties70: return None
    for p in properties70.children_named('P'):
        if p.props and p.props[0] == prop_name:
            return p.props[4:]
    return None

# ----------------------------- math -----------------------------

def norm3(v):
    x,y,z=v; l=math.sqrt(x*x+y*y+z*z)
    if l <= 1e-12: return (0.0,0.0,1.0)
    return (x/l,y/l,z/l)

def euler_xyz_degrees_to_quat(rx,ry,rz):
    # Inverse of the exporter display path: intrinsic XYZ / roll-pitch-yaw convention.
    x=math.radians(rx)*0.5; y=math.radians(ry)*0.5; z=math.radians(rz)*0.5
    cx,sx=math.cos(x),math.sin(x)
    cy,sy=math.cos(y),math.sin(y)
    cz,sz=math.cos(z),math.sin(z)
    qw = cx*cy*cz + sx*sy*sz
    qx = sx*cy*cz - cx*sy*sz
    qy = cx*sy*cz + sx*cy*sz
    qz = cx*cy*sz - sx*sy*cz
    l=math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw) or 1.0
    return (qx/l,qy/l,qz/l,qw/l)

# ----------------------------- skeleton / display list helpers -----------------------------

def parse_shapes_string_table(shape: bytes):
    count=struct.unpack_from('<I', shape, 0x400)[0]
    ptrs=struct.unpack_from('<'+'I'*count, shape, 0x404)
    out=[]
    for p in ptrs:
        off=0x400+p; end=shape.find(b'\0', off)
        out.append(shape[off:end].decode('latin1', errors='replace'))
    return out

def parse_skeleton_records(shape: bytes):
    strings=parse_shapes_string_table(shape)
    SKEL_BASE=0x0EE0; SKEL_ROOT=0x0F20
    records={}
    def rec(off):
        idx,parent,nchild,name_idx=struct.unpack_from('>4i', shape, off)
        q=struct.unpack_from('>4f', shape, off+16)
        t=struct.unpack_from('>3f', shape, off+32)
        child_rels=struct.unpack_from('>'+('I'*nchild), shape, off+48) if nchild else ()
        return {'idx':idx,'parent':parent,'nchild':nchild,'name_idx':name_idx,'name':strings[name_idx],'q':q,'t':t,'off':off,'children':[SKEL_BASE+c for c in child_rels]}
    def walk(off):
        r=rec(off); records[r['idx']]=r
        for c in r['children']: walk(c)
    walk(SKEL_ROOT)
    return records

CMD_QUADS=0x80; CMD_TRIS=0x90; CMD_TRI_STRIP=0x98; CMD_TRI_FAN=0xA0
VALID={CMD_QUADS,CMD_TRIS,CMD_TRI_STRIP,CMD_TRI_FAN}

def read_display_list(shape: bytes, start: int):
    pos=start; faces=[]
    while pos < len(shape)-3 and shape[pos] in VALID:
        op=shape[pos]; count=struct.unpack_from('>H', shape, pos+1)[0]; pos += 3
        verts=[]
        for _ in range(count):
            if pos+6 > len(shape): break
            a,b,c=struct.unpack_from('>3H', shape, pos); pos += 6
            verts.append(a)
        if op==CMD_QUADS:
            for i in range(0,len(verts)-3,4):
                faces.append((verts[i],verts[i+1],verts[i+2])); faces.append((verts[i],verts[i+2],verts[i+3]))
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
    return faces

def build_cp_to_source_map(shape: bytes, manifest: dict):
    cp=[]
    for sm_i,sm in enumerate(manifest['mesh_stats']):
        dl=parse_int_maybe_hex(sm['display_list_start'])
        faces=read_display_list(shape, dl)
        for f in faces:
            for idx in f:
                cp.append((sm_i, idx))
    return cp

# ----------------------------- FBX extraction -----------------------------

def extract_fbx_mesh(path: Path):
    roots,version=parse_fbx(path)
    geoms=[g for g in object_nodes(roots,'Geometry') if len(g.props)>=3 and str(g.props[2]).lower()=='mesh']
    if not geoms: geoms=object_nodes(roots,'Geometry')
    if not geoms: raise ValueError('No FBX Geometry/Mesh node found.')
    geom=geoms[0]
    verts_node=geom.child('Vertices')
    pvi_node=geom.child('PolygonVertexIndex')
    if not verts_node or not pvi_node:
        raise ValueError('FBX mesh lacks Vertices or PolygonVertexIndex.')
    flat=verts_node.props[0]
    vertices=[tuple(map(float,flat[i:i+3])) for i in range(0,len(flat),3)]
    pvi=[int(x) for x in pvi_node.props[0]]
    cp_seq=[(-x-1 if x < 0 else x) for x in pvi]
    # normals
    normal_by_pv=None; normal_by_cp=None
    len_pv=len(cp_seq)
    for le in geom.children_named('LayerElementNormal'):
        normals=le.child('Normals')
        if not normals: continue
        vals=normals.props[0]
        triples=[norm3(tuple(map(float,vals[i:i+3]))) for i in range(0,len(vals),3)]
        mapping=(le.child('MappingInformationType').props[0] if le.child('MappingInformationType') else '')
        ref=(le.child('ReferenceInformationType').props[0] if le.child('ReferenceInformationType') else '')
        idx_node=le.child('NormalsIndex') or le.child('NormalIndex')
        if mapping in ('ByPolygonVertex',''):
            if ref=='IndexToDirect' and idx_node:
                idxs=[int(x) for x in idx_node.props[0]]
                normal_by_pv=[triples[i] for i in idxs]
            else:
                normal_by_pv=triples
        elif mapping in ('ByVertice','ByVertex','ByControlPoint'):
            if ref=='IndexToDirect' and idx_node:
                idxs=[int(x) for x in idx_node.props[0]]
                normal_by_cp=[triples[i] for i in idxs]
            else:
                normal_by_cp=triples
        break
    # UVs
    uv_by_pv=None; uv_by_cp=None
    for le in geom.children_named('LayerElementUV'):
        uvn=le.child('UV')
        if not uvn: continue
        vals=uvn.props[0]
        pairs=[tuple(map(float,vals[i:i+2])) for i in range(0,len(vals),2)]
        mapping=(le.child('MappingInformationType').props[0] if le.child('MappingInformationType') else '')
        ref=(le.child('ReferenceInformationType').props[0] if le.child('ReferenceInformationType') else '')
        idx_node=le.child('UVIndex') or le.child('TextureUVIndex')
        if mapping in ('ByPolygonVertex',''):
            if ref=='IndexToDirect' and idx_node:
                idxs=[int(x) for x in idx_node.props[0]]
                uv_by_pv=[pairs[i] for i in idxs]
            else:
                uv_by_pv=pairs
        elif mapping in ('ByVertice','ByVertex','ByControlPoint'):
            if ref=='IndexToDirect' and idx_node:
                idxs=[int(x) for x in idx_node.props[0]]
                uv_by_cp=[pairs[i] for i in idxs]
            else:
                uv_by_cp=pairs
        break
    # cluster weights
    weights_by_cp=collections.defaultdict(dict)
    for d in object_nodes(roots,'Deformer'):
        if len(d.props) < 3 or str(d.props[2]) != 'Cluster': continue
        cname=clean_fbx_object_name(d.props[1]) if len(d.props)>1 else ''
        bname=cname
        if bname.startswith('Cluster_'): bname=bname[len('Cluster_'):]
        idx_node=d.child('Indexes'); w_node=d.child('Weights')
        if not idx_node or not w_node: continue
        for cp_i,wt in zip(idx_node.props[0], w_node.props[0]):
            if 0 <= int(cp_i) < len(vertices):
                weights_by_cp[int(cp_i)][bname]=float(wt)
    # bone model rest poses
    bone_models={}
    for m in object_nodes(roots,'Model'):
        if len(m.props)<3: continue
        if str(m.props[2]) not in ('LimbNode','Null'): continue
        name=clean_fbx_object_name(m.props[1])
        props=m.child('Properties70')
        t=p_values(props,'Lcl Translation')
        r=p_values(props,'Lcl Rotation')
        if t or r:
            bone_models[name]={'translation':tuple(map(float,t[:3])) if t and len(t)>=3 else None,
                               'rotation_euler_xyz_deg':tuple(map(float,r[:3])) if r and len(r)>=3 else None}
    return {
        'roots': roots, 'version': version, 'vertices': vertices, 'polygon_cp_sequence': cp_seq,
        'normal_by_pv': normal_by_pv, 'normal_by_cp': normal_by_cp,
        'uv_by_pv': uv_by_pv, 'uv_by_cp': uv_by_cp,
        'weights_by_cp': weights_by_cp, 'bone_models': bone_models,
    }

# ----------------------------- patchers -----------------------------

def texture_changed(extracted: Path, manifest: dict, rel: str) -> bool:
    hashes=manifest.get('file_hashes') or {}
    old=hashes.get(rel.replace('\\','/'))
    p=extracted / rel
    if not p.exists(): return False
    if not old: return True
    return sha256_file(p) != old

def patch_textures(shape: bytearray, extracted: Path, manifest: dict, report: dict, patch_unchanged=False):
    for spec in manifest.get('texture_specs', []):
        tex_name=spec['filename']; tex_path=extracted/'textures'/tex_name
        if not tex_path.exists():
            report['texture_patches'].append({'texture':tex_name,'status':'skipped_missing_png'}); continue
        if not patch_unchanged and not texture_changed(extracted, manifest, f'textures/{tex_name}'):
            report['texture_patches'].append({'texture':tex_name,'status':'unchanged_not_patched'}); continue
        fmt=spec['format']; w=int(spec['width']); h=int(spec['height']); off=parse_int_maybe_hex(spec['absolute_offset'])
        try:
            if fmt == 'CMPR': payload=encode_cmpr_png(tex_path,w,h)
            elif fmt == 'RGB565': payload=encode_rgb565_png(tex_path,w,h)
            elif fmt == 'I8': payload=encode_i8_png(tex_path,w,h)
            else:
                report['texture_patches'].append({'texture':tex_name,'format':fmt,'status':'skipped_encoder_not_implemented'}); continue
            expected={'CMPR':w*h//2,'RGB565':w*h*2,'I8':w*h}[fmt]
            if len(payload) != expected: raise ValueError(f'encoded size {len(payload)} != expected {expected}')
            shape[off:off+expected]=payload
            report['texture_patches'].append({'texture':tex_name,'format':fmt,'offset':hex(off),'bytes':expected,'status':'patched'})
        except Exception as e:
            report['texture_patches'].append({'texture':tex_name,'format':fmt,'status':f'error: {e}'})

def choose_top_weights(weight_map: dict, bone_name_to_id: dict, max_count: int):
    out=[]
    for name,wt in weight_map.items():
        if name in bone_name_to_id and wt > 1e-7:
            out.append((bone_name_to_id[name], float(wt)))
    out.sort(key=lambda x: x[1], reverse=True)
    out=out[:max_count]
    s=sum(w for _,w in out)
    if s <= 1e-8: return []
    return [(b,w/s) for b,w in out]

def patch_mesh_from_fbx(shape: bytearray, extracted: Path, manifest: dict, report: dict, patch_unchanged=False):
    fbx_path=extracted / manifest.get('fbx','Godzilla2K.fbx')
    if not fbx_path.exists():
        report['mesh_patch']={'status':'skipped_missing_fbx'}; return
    old=(manifest.get('file_hashes') or {}).get(manifest.get('fbx','Godzilla2K.fbx'))
    if old and sha256_file(fbx_path)==old and not patch_unchanged:
        report['mesh_patch']={'status':'unchanged_not_patched'}; return
    fbx=extract_fbx_mesh(fbx_path)
    cp_map=build_cp_to_source_map(bytes(shape), manifest)
    cp_seq=fbx['polygon_cp_sequence']
    if len(cp_seq) != len(cp_map):
        report['mesh_patch']={'status':'skipped_topology_changed','fbx_polygon_vertices':len(cp_seq),'expected_polygon_vertices':len(cp_map)}; return
    # aggregate per original source vertex
    accum=collections.defaultdict(lambda: {'pos':[0.0,0.0,0.0], 'norm':[0.0,0.0,0.0], 'uv':[0.0,0.0], 'n':0, 'weights':collections.defaultdict(float), 'wn':0})
    verts=fbx['vertices']
    for pv_i,(sm_i,src_i) in enumerate(cp_map):
        cp_i=cp_seq[pv_i]
        if cp_i < 0 or cp_i >= len(verts):
            report['mesh_patch']={'status':'skipped_invalid_polygon_index','index':cp_i}; return
        key=(sm_i,src_i); a=accum[key]
        p=verts[cp_i]
        a['pos'][0]+=p[0]; a['pos'][1]+=p[1]; a['pos'][2]+=p[2]
        if fbx['normal_by_pv'] and pv_i < len(fbx['normal_by_pv']): n=fbx['normal_by_pv'][pv_i]
        elif fbx['normal_by_cp'] and cp_i < len(fbx['normal_by_cp']): n=fbx['normal_by_cp'][cp_i]
        else: n=None
        if n:
            a['norm'][0]+=n[0]; a['norm'][1]+=n[1]; a['norm'][2]+=n[2]
        if fbx['uv_by_pv'] and pv_i < len(fbx['uv_by_pv']): uv=fbx['uv_by_pv'][pv_i]
        elif fbx['uv_by_cp'] and cp_i < len(fbx['uv_by_cp']): uv=fbx['uv_by_cp'][cp_i]
        else: uv=None
        if uv:
            a['uv'][0]+=uv[0]; a['uv'][1]+=uv[1]
        wmap=fbx['weights_by_cp'].get(cp_i)
        if wmap:
            for bn,wt in wmap.items(): a['weights'][bn]+=wt
            a['wn'] += 1
        a['n'] += 1
    bones=manifest.get('bones',[])
    bone_name_to_id={b['name']:int(b['idx']) for b in bones}
    # Also allow Cluster names that use underscores instead of spaces.
    for b in bones:
        bone_name_to_id.setdefault(str(b['name']).replace(' ','_'), int(b['idx']))
    patched=0; weights_patched=0; reduced=0
    submeshes=manifest['mesh_stats']
    for (sm_i,src_i),a in accum.items():
        n=max(1,a['n'])
        sm=submeshes[sm_i]
        stride=parse_int_maybe_hex(sm['vertex_stride'])
        off=parse_int_maybe_hex(sm['vertex_start']) + src_i*stride
        # Map stride -> layout. Must match utils/bdg_to_fbx_extract_all and
        # main._read_pos_uv_nrm. Without explicit blend52/blend60/skin48
        # branches, those streams used to be patched as if they were skin64,
        # writing UVs and weights into the wrong byte offsets and corrupting
        # JetJaguar/Biollante/Gigan/SpaceGodzilla on save.
        layout={64:'skin64',76:'blend76',52:'blend52',60:'blend60',48:'skin48',40:'skin40'}.get(stride,'skin64')
        pos=(a['pos'][0]/n,a['pos'][1]/n,a['pos'][2]/n)
        norm=norm3((a['norm'][0]/n,a['norm'][1]/n,a['norm'][2]/n)) if any(abs(x)>1e-8 for x in a['norm']) else None
        uv=(a['uv'][0]/n,a['uv'][1]/n) if any(abs(x)>1e-8 for x in a['uv']) else None
        struct.pack_into('>3f', shape, off, *pos)
        if layout == 'skin64':
            if uv: struct.pack_into('>2f', shape, off+20, float(uv[0]), float(1.0-uv[1]))
            if norm: struct.pack_into('>3f', shape, off+28, *norm)
            if a['wn']:
                wavg={bn:wt/max(1,a['wn']) for bn,wt in a['weights'].items()}
                top=choose_top_weights(wavg, bone_name_to_id, 2)
                if top:
                    if len(wavg)>2: reduced+=1
                    if len(top)==1:
                        b0=top[0][0]; b1=b0; w0=1.0
                    else:
                        b0,w0=top[0]; b1,w1=top[1]
                    struct.pack_into('>f', shape, off+12, float(w0))
                    struct.pack_into('>2H', shape, off+16, int(b0), int(b1))
                    weights_patched+=1
        elif layout in ('skin40','skin48'):
            if norm: struct.pack_into('>3f', shape, off+20, *norm)
            if uv: struct.pack_into('>2f', shape, off+32, float(uv[0]), float(1.0-uv[1]))
            if a['wn']:
                wavg={bn:wt/max(1,a['wn']) for bn,wt in a['weights'].items()}
                top=choose_top_weights(wavg, bone_name_to_id, 2)
                if top:
                    if len(wavg)>2: reduced+=1
                    if len(top)==1:
                        b0=top[0][0]; b1=b0; w0=1.0
                    else:
                        b0,w0=top[0]; b1,w1=top[1]
                    struct.pack_into('>f', shape, off+12, float(w0))
                    struct.pack_into('>2H', shape, off+16, int(b0), int(b1))
                    weights_patched+=1
        elif layout in ('blend52','blend60'):
            if norm: struct.pack_into('>3f', shape, off+32, *norm)
            if uv: struct.pack_into('>2f', shape, off+44, float(uv[0]), float(1.0-uv[1]))
            if a['wn']:
                wavg={bn:wt/max(1,a['wn']) for bn,wt in a['weights'].items()}
                top=choose_top_weights(wavg, bone_name_to_id, 4)
                if top:
                    if len(wavg)>4: reduced+=1
                    while len(top)<4: top.append((top[-1][0],0.0))
                    total=sum(w for _,w in top) or 1.0
                    top=[(b,w/total) for b,w in top]
                    struct.pack_into('>3f', shape, off+12, float(top[0][1]), float(top[1][1]), float(top[2][1]))
                    struct.pack_into('>4H', shape, off+24, int(top[0][0]), int(top[1][0]), int(top[2][0]), int(top[3][0]))
                    weights_patched+=1
        else:  # blend76
            if uv: struct.pack_into('>2f', shape, off+32, float(uv[0]), float(1.0-uv[1]))
            if norm: struct.pack_into('>3f', shape, off+40, *norm)
            if a['wn']:
                wavg={bn:wt/max(1,a['wn']) for bn,wt in a['weights'].items()}
                top=choose_top_weights(wavg, bone_name_to_id, 4)
                if top:
                    if len(wavg)>4: reduced+=1
                    while len(top)<4: top.append((top[-1][0],0.0))
                    total=sum(w for _,w in top) or 1.0
                    top=[(b,w/total) for b,w in top]
                    struct.pack_into('>3f', shape, off+12, float(top[0][1]), float(top[1][1]), float(top[2][1]))
                    struct.pack_into('>4H', shape, off+24, int(top[0][0]), int(top[1][0]), int(top[2][0]), int(top[3][0]))
                    weights_patched+=1
        patched += 1
    report['mesh_patch']={'status':'patched','source_vertices_patched':patched,'weights_patched':weights_patched,'weights_reduced_to_source_limits':reduced,'fbx_version':fbx['version']}

def patch_skeleton_from_fbx(shape: bytearray, extracted: Path, manifest: dict, report: dict, patch_unchanged=False):
    fbx_path=extracted / manifest.get('fbx','Godzilla2K.fbx')
    if not fbx_path.exists():
        report['skeleton_patch']={'status':'skipped_missing_fbx'}; return
    old=(manifest.get('file_hashes') or {}).get(manifest.get('fbx','Godzilla2K.fbx'))
    if old and sha256_file(fbx_path)==old and not patch_unchanged:
        report['skeleton_patch']={'status':'unchanged_not_patched'}; return
    fbx=extract_fbx_mesh(fbx_path)
    records=parse_skeleton_records(bytes(shape))
    # Flexible name map: exact, space->underscore, trailing clean names.
    models=fbx['bone_models']
    patched=0; missing=[]
    for idx,r in records.items():
        name=r['name']
        cand=models.get(name) or models.get(name.replace(' ','_'))
        if not cand:
            # trailing match fallback
            for k,v in models.items():
                if k.endswith(name) or k.endswith(name.replace(' ','_')):
                    cand=v; break
        if not cand:
            missing.append(name); continue
        off=r['off']
        if cand.get('rotation_euler_xyz_deg'):
            q=euler_xyz_degrees_to_quat(*cand['rotation_euler_xyz_deg'])
            struct.pack_into('>4f', shape, off+16, *q)
        if cand.get('translation'):
            struct.pack_into('>3f', shape, off+32, *cand['translation'])
        patched += 1
    report['skeleton_patch']={'status':'patched' if patched else 'skipped_no_matching_bones','bones_patched':patched,'bones_missing_from_fbx':missing[:20],'missing_count':len(missing)}

def raw_anims_changed(extracted: Path, manifest: dict, name: str) -> bool:
    hashes=manifest.get('file_hashes') or {}
    rel=f'animations_raw/{name}.bin'
    p=extracted/rel
    if not p.exists(): return False
    old=hashes.get(rel)
    if not old: return True
    return sha256_file(p) != old

def patch_raw_anims(anim: bytearray, extracted: Path, manifest: dict, report: dict, patch_unchanged=False):
    raw_dir=extracted/'animations_raw'
    if not raw_dir.exists():
        report['animation_resource_patches'].append({'status':'skipped_no_animations_raw_folder'}); return
    for res in manifest.get('animation_resource_locations', []):
        raw=raw_dir/f"{res['name']}.bin"
        if not raw.exists():
            report['animation_resource_patches'].append({'clip':res['name'],'status':'skipped_missing_raw_bin'}); continue
        if not patch_unchanged and not raw_anims_changed(extracted,manifest,res['name']):
            report['animation_resource_patches'].append({'clip':res['name'],'status':'unchanged_not_patched'}); continue
        payload=raw.read_bytes(); expected=int(res['size']); off=parse_int_maybe_hex(res['absolute_offset'])
        if len(payload)!=expected:
            report['animation_resource_patches'].append({'clip':res['name'],'status':f'skipped_size_changed_{len(payload)}_expected_{expected}'}); continue
        anim[off:off+expected]=payload
        report['animation_resource_patches'].append({'clip':res['name'],'offset':hex(off),'bytes':expected,'status':'patched_same_size_raw'})

# ----------------------------- main -----------------------------

def main() -> int:
    ap=argparse.ArgumentParser(description='Import an extracted kaiju folder back into copied BDG/PVM files. Current full writeback profile: same-topology Godzilla2K.')
    ap.add_argument('folder', nargs='?', default='.', help='Folder containing one *-Kaiju-Extracted folder and original BDG/PVM files')
    ap.add_argument('--force', action='store_true', help='Overwrite existing *-Kaiju-Reimported folder')
    ap.add_argument('--copy-only', action='store_true', help='Only copy original files, do not patch anything')
    ap.add_argument('--patch-unchanged', action='store_true', help='Patch even files that match extraction hashes')
    ap.add_argument('--no-textures', action='store_true', help='Do not import PNG textures')
    ap.add_argument('--no-fbx', action='store_true', help='Do not import FBX mesh/skeleton rest-pose edits')
    ap.add_argument('--no-raw-anims', action='store_true', help='Do not import same-size animations_raw/*.bin swaps')
    args=ap.parse_args()

    root=Path(clean_windows_folder_arg(args.folder)).resolve()
    extracted=find_extract_folder(root)
    manifest=load_manifest(extracted)
    base=extracted.name[:-len('-Kaiju-Extracted')]
    out=root/f'{base}-Kaiju-Reimported'
    if out.exists():
        if not args.force: raise SystemExit(f'Reimport output already exists: {out}. Delete it or run Import.bat --force.')
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shape_src=find_case_insensitive(root, manifest.get('source',''))
    anim_src=find_case_insensitive(root, manifest.get('animation_source',''))
    if shape_src is None or anim_src is None:
        raise SystemExit(f'Missing original BDGs beside Import.bat: {manifest.get("source")}, {manifest.get("animation_source")}')
    staged_shape=out/shape_src.name; staged_anim=out/anim_src.name
    shutil.copy2(shape_src, staged_shape); shutil.copy2(anim_src, staged_anim)
    staged_pvms=[]
    for pvm_name in manifest.get('pvms',[]):
        p=find_case_insensitive(root,pvm_name)
        if p:
            dst=out/p.name; shutil.copy2(p,dst); staged_pvms.append(dst.name)

    report={
        'extracted_folder': extracted.name,
        'output_folder': out.name,
        'copied': [staged_shape.name, staged_anim.name]+staged_pvms,
        'texture_patches': [], 'animation_resource_patches': [],
        'mesh_patch': {'status':'not_run'}, 'skeleton_patch': {'status':'not_run'},
        'limits': [
            'Requires same FBX polygon order/topology as the extracted mesh for mesh writeback.',
            'Skin64 vertices can store only two influences; blend76 vertices can store only four. Extra FBX weights are reduced to source format limits and reported.',
            'Blender animation curves are not yet re-encoded to proprietary BDG animation channels; use animations_raw same-size binary replacement for now.',
            'PVM files are copied/preserved; this Godzilla profile stores the decoded texture payloads in Shapes.BDG.',
        ],
    }

    if not args.copy_only:
        shape=bytearray(staged_shape.read_bytes())
        anim=bytearray(staged_anim.read_bytes())
        if not args.no_textures:
            patch_textures(shape, extracted, manifest, report, patch_unchanged=args.patch_unchanged)
        if not args.no_fbx:
            try:
                patch_mesh_from_fbx(shape, extracted, manifest, report, patch_unchanged=args.patch_unchanged)
            except Exception as e:
                report['mesh_patch']={'status':f'error: {type(e).__name__}: {e}'}
            try:
                patch_skeleton_from_fbx(shape, extracted, manifest, report, patch_unchanged=args.patch_unchanged)
            except Exception as e:
                report['skeleton_patch']={'status':f'error: {type(e).__name__}: {e}'}
        if not args.no_raw_anims:
            patch_raw_anims(anim, extracted, manifest, report, patch_unchanged=args.patch_unchanged)
        staged_shape.write_bytes(shape)
        staged_anim.write_bytes(anim)

    (out/'import_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Created reimported folder: {out}')
    print(f'  Shapes BDG: {staged_shape.name}')
    print(f'  Animation BDG: {staged_anim.name}')
    if staged_pvms: print('  PVMs: ' + ', '.join(staged_pvms))
    print('Wrote import_report.json')
    print('Patch summary:')
    print('  textures:', collections.Counter(x.get('status','unknown') for x in report['texture_patches']))
    print('  mesh:', report['mesh_patch'].get('status'))
    print('  skeleton:', report['skeleton_patch'].get('status'))
    print('  raw anims:', collections.Counter(x.get('status','unknown') for x in report['animation_resource_patches']))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
