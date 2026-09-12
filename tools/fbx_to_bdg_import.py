from __future__ import annotations
from pathlib import Path
import argparse, collections, hashlib, json, math, os, shutil, struct, sys, tempfile, zlib
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
    m = extracted / 'import_log.json'
    if not m.exists():
        raise SystemExit(f'Missing import_log.json in {extracted}')
    return json.loads(m.read_text(encoding='utf-8'))

# ----------------------------- Wii texture encoders -----------------------------

def rgb565_pack(r:int,g:int,b:int) -> int:
    r5=max(0,min(31,round(r*31/255)))
    g6=max(0,min(63,round(g*63/255)))
    b5=max(0,min(31,round(b*31/255)))
    return (r5<<11)|(g6<<5)|b5

def rgb565_unpack(c:int):
    return (((c>>11)&31)*255//31, ((c>>5)&63)*255//63, (c&31)*255//31, 255)

def rgb5a3_pack(r:int,g:int,b:int,a:int) -> int:
    if a < 224:
        a3=max(0,min(7,round(a*7/255)))
        r4=max(0,min(15,round(r*15/255)))
        g4=max(0,min(15,round(g*15/255)))
        b4=max(0,min(15,round(b*15/255)))
        return (a3<<12)|(r4<<8)|(g4<<4)|b4
    r5=max(0,min(31,round(r*31/255)))
    g5=max(0,min(31,round(g*31/255)))
    b5=max(0,min(31,round(b*31/255)))
    return 0x8000|(r5<<10)|(g5<<5)|b5

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

def encode_rgb5a3_png(path: Path, width: int, height: int) -> bytes:
    img = Image.open(path).convert('RGBA').resize((width, height))
    pix = img.load(); out=bytearray()
    # Wii RGB5A3 texture tiles are 4x4 pixels.
    for ty in range(0,height,4):
      for tx in range(0,width,4):
        for y in range(4):
          for x in range(4):
            xx,yy=tx+x,ty+y
            r,g,b,a = pix[xx,yy] if xx<width and yy<height else (0,0,0,0)
            out += struct.pack('>H', rgb5a3_pack(r,g,b,a))
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

def encode_ia4_png(path: Path, width: int, height: int) -> bytes:
    img = Image.open(path).convert('RGBA').resize((width, height))
    pix = img.load(); out=bytearray()
    for ty in range(0,height,4):
      for tx in range(0,width,8):
        for y in range(4):
          for x in range(8):
            xx,yy=tx+x,ty+y
            if xx<width and yy<height:
                r,g,b,a = pix[xx,yy]
                i = round((r*0.299 + g*0.587 + b*0.114) * 15 / 255)
                an = round(a * 15 / 255)
            else:
                i = 0; an = 0
            out.append((max(0,min(15,an)) << 4) | max(0,min(15,i)))
    return bytes(out)

def encode_ia8_png(path: Path, width: int, height: int) -> bytes:
    img = Image.open(path).convert('RGBA').resize((width, height))
    pix = img.load(); out=bytearray()
    for ty in range(0,height,4):
      for tx in range(0,width,4):
        for y in range(4):
          for x in range(4):
            xx,yy=tx+x,ty+y
            if xx<width and yy<height:
                r,g,b,a = pix[xx,yy]
                i = round(r*0.299 + g*0.587 + b*0.114)
            else:
                i = 0; a = 0
            out.append(max(0,min(255,a)))
            out.append(max(0,min(255,i)))
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

def texture_level_size(fmt: str, width: int, height: int) -> int:
    if fmt == 'CMPR':
        return max(8, width) * max(8, height) // 2
    if fmt in ('RGB565', 'RGB5A3', 'IA8'):
        return width * height * 2
    if fmt in ('I8', 'IA4'):
        return width * height
    raise ValueError(f'unsupported texture format {fmt}')

def encode_texture_png(path: Path, fmt: str, width: int, height: int) -> bytes:
    if fmt == 'CMPR': return encode_cmpr_png(path,width,height)
    if fmt == 'RGB565': return encode_rgb565_png(path,width,height)
    if fmt == 'RGB5A3': return encode_rgb5a3_png(path,width,height)
    if fmt == 'I8': return encode_i8_png(path,width,height)
    if fmt == 'IA4': return encode_ia4_png(path,width,height)
    if fmt == 'IA8': return encode_ia8_png(path,width,height)
    raise ValueError(f'unsupported texture format {fmt}')

def encode_texture_mip_chain_png(path: Path, fmt: str, width: int, height: int, mip_count: int, encoded_size: int | None = None) -> bytes:
    out=bytearray()
    mip_count=max(1,int(mip_count or 1))
    for _ in range(mip_count):
        payload=encode_texture_png(path,fmt,width,height)
        expected=texture_level_size(fmt,width,height)
        if len(payload) != expected:
            raise ValueError(f'encoded mip {width}x{height} size {len(payload)} != expected {expected}')
        out += payload
        width=max(1,width//2); height=max(1,height//2)
    if encoded_size is not None and len(out) != encoded_size:
        raise ValueError(f'encoded mip chain size {len(out)} != expected resource size {encoded_size}')
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
    return s.strip()

def p_values(properties70: FbxNode, prop_name: str):
    if not properties70: return None
    for p in properties70.children_named('P'):
        if p.props and p.props[0] == prop_name:
            return p.props[4:]
    return None

def bdg_bone_name_variants(name: str):
    name=str(name).strip()
    out=[]
    def add(v):
        if v and v not in out:
            out.append(v)
    add(name)
    add(name.replace(' ', '_'))
    add(name + 'Model')
    add(name.replace(' ', '_') + 'Model')
    return out

def find_bdg_bone_model(models: dict, name: str):
    for variant in bdg_bone_name_variants(name):
        cand=models.get(variant)
        if cand:
            return cand
    for k,v in models.items():
        if any(k.endswith(variant) for variant in bdg_bone_name_variants(name)):
            return v
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

def parse_shapes_string_table(shape: bytes, offset=0x400):
    if offset < 0 or offset + 4 > len(shape):
        raise ValueError(f'Invalid Shapes.BDG string table offset: {offset:#x}')
    count=struct.unpack_from('<I', shape, offset)[0]
    if count <= 0 or count > 65535 or offset + 4 + count*4 > len(shape):
        raise ValueError(f'Invalid Shapes.BDG string count at {offset:#x}: {count}')
    ptrs=struct.unpack_from('<'+'I'*count, shape, offset+4)
    out=[]
    for p in ptrs:
        off=offset+p
        if off < offset or off >= len(shape):
            raise ValueError(f'Invalid Shapes.BDG string pointer: {p:#x}')
        end=shape.find(b'\0', off)
        if end < 0:
            raise ValueError(f'Unterminated Shapes.BDG string at {off:#x}')
        out.append(shape[off:end].decode('latin1', errors='replace'))
    return out

def parse_skeleton_records(shape: bytes, manifest: dict | None = None):
    manifest=manifest or {}
    string_table_offset=parse_int_maybe_hex(manifest.get('string_table_offset', 0x400))
    strings=parse_shapes_string_table(shape, string_table_offset)
    skel_base=parse_int_maybe_hex(manifest.get('skeleton_base', 0x0EE0))
    skel_root=parse_int_maybe_hex(manifest.get('skeleton_root', 0x0F20))
    records={}
    def rec(off):
        if off < 0 or off + 48 > len(shape):
            raise ValueError(f'Skeleton record is outside Shapes.BDG: {off:#x}')
        idx,parent,nchild,name_idx=struct.unpack_from('>4i', shape, off)
        if not (0 <= idx < len(strings)):
            raise ValueError(f'Invalid skeleton bone index {idx} at {off:#x}')
        if not (-1 <= parent < len(strings)):
            raise ValueError(f'Invalid skeleton parent index {parent} at {off:#x}')
        if not (0 <= nchild < len(strings)):
            raise ValueError(f'Invalid skeleton child count {nchild} at {off:#x}')
        if not (0 <= name_idx < len(strings)):
            raise ValueError(f'Invalid skeleton name index {name_idx} at {off:#x}')
        if off + 48 + nchild*4 > len(shape):
            raise ValueError(f'Skeleton child table is outside Shapes.BDG at {off:#x}')
        q=struct.unpack_from('>4f', shape, off+16)
        t=struct.unpack_from('>3f', shape, off+32)
        child_rels=struct.unpack_from('>'+('I'*nchild), shape, off+48) if nchild else ()
        children=[skel_base+c for c in child_rels]
        if any(c < 0 or c + 48 > len(shape) for c in children):
            raise ValueError(f'Invalid skeleton child pointer at {off:#x}')
        return {'idx':idx,'parent':parent,'nchild':nchild,'name_idx':name_idx,'name':strings[name_idx],'q':q,'t':t,'off':off,'children':children}
    def walk(off):
        if off in {r['off'] for r in records.values()}:
            return
        r=rec(off); records[r['idx']]=r
        for c in r['children']: walk(c)
    walk(skel_root)
    return records

def _qmat(q):
    x,y,z,w=q
    n=x*x+y*y+z*z+w*w
    if n < 1e-12:
        return [[1,0,0],[0,1,0],[0,0,1]]
    s=2/n
    xx,yy,zz=x*x*s,y*y*s,z*z*s
    xy,xz,yz=x*y*s,x*z*s,y*z*s
    wx,wy,wz=w*x*s,w*y*s,w*z*s
    return [[1-yy-zz,xy-wz,xz+wy],[xy+wz,1-xx-zz,yz-wx],[xz-wy,yz+wx,1-xx-yy]]

def _mm(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def _local_matrix(record):
    r=_qmat(record['q'])
    x,y,z=record['t']
    return [[r[0][0],r[0][1],r[0][2],x],[r[1][0],r[1][1],r[1][2],y],[r[2][0],r[2][1],r[2][2],z],[0,0,0,1]]

def _global_positions_from_records(records: dict[int, dict]):
    mats={}
    def mat_for(idx):
        if idx in mats:
            return mats[idx]
        rec=records[idx]
        parent=int(rec.get('parent', -1))
        local=_local_matrix(rec)
        mats[idx]=_mm(mat_for(parent), local) if parent in records else local
        return mats[idx]
    out={}
    for idx in records:
        m=mat_for(idx)
        out[idx]=(float(m[0][3]), float(m[1][3]), float(m[2][3]))
    return out

def _bdg_vertex_weights(shape: bytes | bytearray, off: int, layout: str):
    try:
        if layout in ('skin64','skin48','skin40'):
            w0=struct.unpack_from('>f', shape, off+12)[0]
            b0,b1=struct.unpack_from('>2H', shape, off+16)
            w0=max(0.0,min(1.0,float(w0)))
            if b0 == b1:
                return [(int(b0),1.0)]
            return [(int(b0),w0),(int(b1),1.0-w0)]
        w0,w1,w2=struct.unpack_from('>3f', shape, off+12)
        bones=struct.unpack_from('>4H', shape, off+24)
        vals=[max(0.0,float(w0)),max(0.0,float(w1)),max(0.0,float(w2))]
        vals.append(max(0.0,1.0-sum(vals)))
        total=sum(vals)
        if total <= 1e-8:
            return []
        return [(int(b),v/total) for b,v in zip(bones, vals) if v > 1e-6]
    except Exception:
        return []

def patch_mesh_for_skeleton_position_edits(shape: bytearray, manifest: dict, report: dict):
    skel=report.get('skeleton_patch') or {}
    changed=skel.get('changed_bones') or []
    if not changed:
        report['mesh_skeleton_bake']={'status':'skipped_no_changed_bones'}
        return
    old_globals={}
    for bone in manifest.get('bones', []):
        if 'idx' in bone and bone.get('global_position') is not None:
            old_globals[int(bone['idx'])]=tuple(float(v) for v in bone['global_position'])
    if not old_globals:
        report['mesh_skeleton_bake']={'status':'skipped_missing_original_globals'}
        return
    new_records=parse_skeleton_records(bytes(shape), manifest)
    new_globals=_global_positions_from_records(new_records)
    deltas={}
    for idx,old in old_globals.items():
        new=new_globals.get(idx)
        if not new:
            continue
        delta=(new[0]-old[0], new[1]-old[1], new[2]-old[2])
        if any(abs(v) > 1e-4 for v in delta):
            deltas[idx]=delta
    if not deltas:
        report['mesh_skeleton_bake']={'status':'skipped_no_global_bone_delta'}
        return
    moved=0
    max_delta=0.0
    for sm in manifest.get('mesh_stats', []):
        count=int(sm.get('vertex_count') or 0)
        stride=parse_int_maybe_hex(sm['vertex_stride'])
        start=parse_int_maybe_hex(sm['vertex_start'])
        layout={64:'skin64',76:'blend76',52:'blend52',60:'blend60',48:'skin48',40:'skin40'}.get(stride,'skin64')
        for src_i in range(count):
            off=start + src_i*stride
            weights=_bdg_vertex_weights(shape, off, layout)
            if not weights:
                continue
            dx=dy=dz=0.0
            for bone,weight in weights:
                delta=deltas.get(int(bone))
                if not delta:
                    continue
                dx += delta[0]*weight
                dy += delta[1]*weight
                dz += delta[2]*weight
            mag=math.sqrt(dx*dx+dy*dy+dz*dz)
            if mag <= 1e-4:
                continue
            x,y,z=struct.unpack_from('>3f', shape, off)
            struct.pack_into('>3f', shape, off, x+dx, y+dy, z+dz)
            moved += 1
            max_delta=max(max_delta, mag)
    report['mesh_skeleton_bake']={
        'status':'patched' if moved else 'skipped_no_weighted_vertices',
        'vertices_moved':moved,
        'bones_with_global_delta':len(deltas),
        'max_weighted_delta':max_delta,
    }

CMD_QUADS=0x80; CMD_TRIS=0x90; CMD_TRI_STRIP=0x98; CMD_TRI_FAN=0xA0
VALID={CMD_QUADS,CMD_TRIS,CMD_TRI_STRIP,CMD_TRI_FAN}

def read_display_list(shape: bytes, start: int, index_width: int = 6):
    pos=start; faces=[]
    cmds=0
    while pos < len(shape)-3:
        if shape[pos] == 0x00 and cmds > 0:
            pos += 1
            continue
        if shape[pos] not in VALID:
            break
        op=shape[pos]; count=struct.unpack_from('>H', shape, pos+1)[0]; pos += 3
        if count < 3 or count > 4096 or pos + index_width * count > len(shape):
            break
        verts=[]
        for _ in range(count):
            if index_width == 6:
                a,b,c=struct.unpack_from('>3H', shape, pos); pos += 6
            elif index_width == 3:
                a,b,c=shape[pos],shape[pos+1],shape[pos+2]; pos += 3
            elif index_width == 4:
                a,b,c=shape[pos],shape[pos+1],shape[pos+2]; pos += 4
            elif index_width == 8:
                a,b,c,_d=struct.unpack_from('>4H', shape, pos); pos += 8
            else:
                raise ValueError(f'unsupported display-list index width {index_width}')
            verts.append(a)
        cmds += 1
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

def _duplicate_seam_face_drops(shape: bytes, manifest: dict, cp: list[tuple[int,int]]):
    expected_drops=int(manifest.get('duplicate_seam_faces_skipped') or 0)
    if expected_drops <= 0 or len(cp)%3:
        return set()
    groups=collections.defaultdict(list)
    face_count=len(cp)//3
    for face_i in range(face_count):
        positions=[]; uvs=[]
        for sm_i,src_i in cp[face_i*3:face_i*3+3]:
            sm=manifest['mesh_stats'][sm_i]
            stride=parse_int_maybe_hex(sm['vertex_stride'])
            off=parse_int_maybe_hex(sm['vertex_start'])+src_i*stride
            positions.append(struct.unpack_from('>3f',shape,off))
            uvs.append(_native_stored_uv(shape,manifest,sm_i,src_i) or (0.0,0.0))
        key=tuple(sorted((round(float(p[0]),4),round(float(p[1]),4),round(float(p[2]),4)) for p in positions))
        uv_span=max(math.dist(uvs[a],uvs[b]) for a,b in ((0,1),(1,2),(2,0)))
        groups[key].append((face_i,uv_span))
    drop=set()
    for rows in groups.values():
        if len(rows)<2:
            continue
        face_ids=[face_i for face_i,_span in rows]
        if max(face_ids)-min(face_ids)<=64:
            continue
        spans=[span for _face_i,span in rows]
        keep=(min(rows,key=lambda row:row[1]) if max(spans)-min(spans)>0.02 else max(rows,key=lambda row:row[0]))[0]
        drop.update(face_i for face_i,_span in rows if face_i!=keep)
    return drop if len(drop)==expected_drops else set()

def build_cp_to_source_map(shape: bytes, manifest: dict):
    cp=[]
    for sm_i,sm in enumerate(manifest['mesh_stats']):
        dl=parse_int_maybe_hex(sm['display_list_start'])
        faces=read_display_list(shape, dl, int(sm.get('index_width', 6)))
        for f in faces:
            for idx in f:
                cp.append((sm_i, idx))
    drop=_duplicate_seam_face_drops(shape,manifest,cp)
    if drop:
        face_count=len(cp)//3
        cp=[item for face_i in range(face_count) if face_i not in drop for item in cp[face_i*3:face_i*3+3]]
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
    polygon_faces=[]; cur=[]
    for raw_i, cp_i in zip(pvi, cp_seq):
        cur.append(int(cp_i))
        if raw_i < 0:
            polygon_faces.append(cur)
            cur=[]
    if cur:
        polygon_faces.append(cur)
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
        'polygon_faces': polygon_faces,
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

def _texture_specs_for_writeback(shape: bytes | bytearray, manifest: dict):
    specs = list(manifest.get('texture_specs') or [])
    if specs:
        return specs
    textures = list(manifest.get('textures') or [])
    if not textures:
        return []
    abs_by_rid = {}
    try:
        from bdg_to_fbx_extract_all import texture_entries
        entries, data_base = texture_entries(bytes(shape))
        for e in entries:
            abs_by_rid[int(e.get('rid'))] = int(e.get('abs'))
    except Exception:
        data_base = None
    out = []
    for tex in textures:
        try:
            filename = tex.get('file') or tex.get('filename')
            fmt = tex.get('format')
            w = int(tex.get('width'))
            h = int(tex.get('height'))
        except Exception:
            continue
        encoded_size = None
        mip_count = tex.get('mip_count')
        try:
            if tex.get('size') is not None:
                encoded_size = parse_int_maybe_hex(tex.get('size'))
        except Exception:
            encoded_size = None
        off = None
        try:
            rid = int(tex.get('rid'))
            off = abs_by_rid.get(rid)
        except Exception:
            off = None
        if off is None and data_base is not None and tex.get('rel') is not None:
            try:
                off = int(data_base) + parse_int_maybe_hex(tex.get('rel'))
            except Exception:
                off = None
        if filename and fmt and off is not None:
            spec = {'filename': filename, 'format': fmt, 'width': w, 'height': h, 'absolute_offset': off}
            if mip_count is not None:
                spec['mip_count'] = mip_count
            if encoded_size is not None:
                spec['encoded_size'] = encoded_size
            out.append(spec)
    return out

def patch_textures(shape: bytearray, extracted: Path, manifest: dict, report: dict, patch_unchanged=False):
    for spec in _texture_specs_for_writeback(shape, manifest):
        tex_name=spec['filename']; tex_path=extracted/'textures'/tex_name
        if not tex_path.exists():
            report['texture_patches'].append({'texture':tex_name,'status':'skipped_missing_png'}); continue
        if not patch_unchanged and not texture_changed(extracted, manifest, f'textures/{tex_name}'):
            report['texture_patches'].append({'texture':tex_name,'status':'unchanged_not_patched'}); continue
        fmt=spec['format']; w=int(spec['width']); h=int(spec['height']); off=parse_int_maybe_hex(spec['absolute_offset'])
        try:
            if fmt not in ('CMPR','RGB565','RGB5A3','I8','IA4','IA8'):
                report['texture_patches'].append({'texture':tex_name,'format':fmt,'status':'skipped_encoder_not_implemented'}); continue
            mip_count = int(spec.get('mip_count') or 1)
            encoded_size = spec.get('encoded_size')
            expected = int(encoded_size) if encoded_size is not None else sum(
                texture_level_size(fmt, max(1,w>>i), max(1,h>>i)) for i in range(mip_count)
            )
            payload=encode_texture_mip_chain_png(tex_path,fmt,w,h,mip_count,expected)
            if len(payload) != expected: raise ValueError(f'encoded size {len(payload)} != expected {expected}')
            shape[off:off+expected]=payload
            report['texture_patches'].append({'texture':tex_name,'format':fmt,'offset':hex(off),'bytes':expected,'mip_count':mip_count,'status':'patched'})
        except Exception as e:
            report['texture_patches'].append({'texture':tex_name,'format':fmt,'status':f'error: {e}'})

def choose_top_weights(weight_map: dict, bone_name_to_id: dict, max_count: int):
    out=[]
    for name,wt in weight_map.items():
        candidates = [str(name)]
        if str(name).endswith('SubDeformer'):
            candidates.append(str(name)[:-len('SubDeformer')])
        if str(name).startswith('Cluster_'):
            candidates.append(str(name)[len('Cluster_'):])
        bone_id = None
        for cand in candidates:
            if cand in bone_name_to_id:
                bone_id = bone_name_to_id[cand]
                break
            alt = cand.replace('_', ' ')
            if alt in bone_name_to_id:
                bone_id = bone_name_to_id[alt]
                break
        if bone_id is not None and wt > 1e-7:
            out.append((bone_id, float(wt)))
    out.sort(key=lambda x: x[1], reverse=True)
    out=out[:max_count]
    s=sum(w for _,w in out)
    if s <= 1e-8: return []
    return [(b,w/s) for b,w in out]

def _bundle_entries_from_bytes(data: bytes):
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.BDG') as f:
            f.write(data)
            tmp_name = f.name
        from parser_core import PipeworksParser
        parser = PipeworksParser(tmp_name)
        entries = parser.parse()
        return parser, entries
    finally:
        if tmp_name:
            try: os.unlink(tmp_name)
            except Exception: pass

def _bdg_vertex_uv(shape: bytes | bytearray, off: int, layout: str):
    try:
        if layout == 'skin64':
            u, v = struct.unpack_from('>2f', shape, off + 20)
        elif layout in ('skin40', 'skin48'):
            u, v = struct.unpack_from('>2f', shape, off + 32)
        elif layout in ('blend52', 'blend60'):
            u, v = struct.unpack_from('>2f', shape, off + 44)
        else:
            u, v = struct.unpack_from('>2f', shape, off + 32)
    except Exception:
        return None
    if not all(math.isfinite(float(x)) for x in (u, v)):
        return None
    return (float(u), float(1.0 - v))

def _uv_dist2(a, b):
    return (float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2

def _choose_seam_safe_uv(samples, original_uv=None):
    samples = [tuple(map(float, uv)) for uv in samples or [] if uv is not None]
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    spread = 0.0
    for i, uv_a in enumerate(samples):
        for uv_b in samples[i + 1:]:
            spread = max(spread, _uv_dist2(uv_a, uv_b))
    if spread > 0.035:
        if original_uv is not None:
            return min(samples, key=lambda uv: _uv_dist2(uv, original_uv))
        counts = collections.Counter((round(uv[0], 5), round(uv[1], 5)) for uv in samples)
        uv, _count = counts.most_common(1)[0]
        return (float(uv[0]), float(uv[1]))
    return (
        sum(float(uv[0]) for uv in samples) / len(samples),
        sum(float(uv[1]) for uv in samples) / len(samples),
    )

def _native_stored_uv(shape: bytes | bytearray, manifest: dict, sm_i: int, src_i: int):
    try:
        sm = manifest['mesh_stats'][int(sm_i)]
        stride = parse_int_maybe_hex(sm['vertex_stride'])
        off = parse_int_maybe_hex(sm['vertex_start']) + int(src_i) * stride
        layout = {64:'skin64',76:'blend76',52:'blend52',60:'blend60',48:'skin48',40:'skin40'}.get(stride, 'skin64')
        if layout == 'skin64':
            return struct.unpack_from('>2f', shape, off + 20)
        if layout in ('skin40', 'skin48'):
            return struct.unpack_from('>2f', shape, off + 32)
        if layout in ('blend52', 'blend60'):
            return struct.unpack_from('>2f', shape, off + 44)
        return struct.unpack_from('>2f', shape, off + 32)
    except Exception:
        return None

def _native_stored_pos(shape: bytes | bytearray, manifest: dict, sm_i: int, src_i: int):
    try:
        sm = manifest['mesh_stats'][int(sm_i)]
        stride = parse_int_maybe_hex(sm['vertex_stride'])
        off = parse_int_maybe_hex(sm['vertex_start']) + int(src_i) * stride
        return struct.unpack_from('>3f', shape, off)
    except Exception:
        return None

def _closest_uv_key(counts, native_uv):
    if not counts or native_uv is None:
        return None
    return min(
        counts,
        key=lambda k: (
            (float(k[0]) - float(native_uv[0])) ** 2 +
            (float(k[1]) - float(native_uv[1])) ** 2,
            -int(counts[k]),
            k,
        ),
    )

def _uv_from_accum(a: dict, n: int, layout: str | None = None, native_uv=None):
    if not any(abs(x) > 1e-8 for x in a['uv']):
        return None, False
    if a.get('uv_count', 0) > 1:
        counts = a.get('uv_values') or {}
        spread = max(
            float(a.get('uv_max_u', 0.0)) - float(a.get('uv_min_u', 0.0)),
            float(a.get('uv_max_v', 0.0)) - float(a.get('uv_min_v', 0.0)),
        )
        if spread > 0.01:
            key = _closest_uv_key(counts, native_uv)
            if key is not None:
                return (float(key[0]), float(key[1])), False
            if layout == 'blend76' and len(counts) >= 2:
                key = min(counts, key=lambda k: (counts[k], k))
                return (float(key[0]), float(key[1])), False
            if layout in ('skin64', 'skin48', 'skin40') and len(counts) >= 2:
                key = max(counts, key=lambda k: (counts[k], k))
                return (float(key[0]), float(key[1])), False
            return None, True
        if counts:
            key = _closest_uv_key(counts, native_uv)
            if key is not None:
                return (float(key[0]), float(key[1])), False
            key = max(counts, key=lambda k: (counts[k], k))
            return (float(key[0]), float(key[1])), False
    return (a['uv'][0] / n, a['uv'][1] / n), False

def _pos_from_accum(a: dict, n: int, fbx_export_scale: float, native_pos=None):
    counts = a.get('pos_values') or {}
    if len(counts) >= 2:
        spread = max(
            float(a.get('pos_max_x', 0.0)) - float(a.get('pos_min_x', 0.0)),
            float(a.get('pos_max_y', 0.0)) - float(a.get('pos_min_y', 0.0)),
            float(a.get('pos_max_z', 0.0)) - float(a.get('pos_min_z', 0.0)),
        ) / fbx_export_scale
        if spread > 2.0:
            key = max(counts, key=lambda k: (counts[k], k))
            return (
                float(key[0]) / fbx_export_scale,
                float(key[1]) / fbx_export_scale,
                float(key[2]) / fbx_export_scale,
            ), True
    return (
        a['pos'][0] / n / fbx_export_scale,
        a['pos'][1] / n / fbx_export_scale,
        a['pos'][2] / n / fbx_export_scale,
    ), False

def _add_pos_to_accum(a: dict, p):
    x = float(p[0])
    y = float(p[1])
    z = float(p[2])
    a['pos'][0] += x
    a['pos'][1] += y
    a['pos'][2] += z
    a['pos_min_x'] = min(float(a.get('pos_min_x', x)), x)
    a['pos_max_x'] = max(float(a.get('pos_max_x', x)), x)
    a['pos_min_y'] = min(float(a.get('pos_min_y', y)), y)
    a['pos_max_y'] = max(float(a.get('pos_max_y', y)), y)
    a['pos_min_z'] = min(float(a.get('pos_min_z', z)), z)
    a['pos_max_z'] = max(float(a.get('pos_max_z', z)), z)
    counts = a.setdefault('pos_values', collections.Counter())
    counts[(round(x, 5), round(y, 5), round(z, 5))] += 1

def _add_uv_to_accum(a: dict, uv):
    if not uv:
        return
    u = float(uv[0])
    v = float(uv[1])
    a['uv'][0] += u
    a['uv'][1] += v
    a['uv_count'] = int(a.get('uv_count', 0)) + 1
    a['uv_min_u'] = min(float(a.get('uv_min_u', u)), u)
    a['uv_max_u'] = max(float(a.get('uv_max_u', u)), u)
    a['uv_min_v'] = min(float(a.get('uv_min_v', v)), v)
    a['uv_max_v'] = max(float(a.get('uv_max_v', v)), v)
    counts = a.setdefault('uv_values', collections.Counter())
    counts[(round(u, 6), round(v, 6))] += 1

def _patch_existing_mesh_vertices(shape: bytearray, fbx: dict, manifest: dict, cp_map: list[tuple[int, int]], allowed_submeshes: set[int] | None = None, pv_index_map: list[int] | None = None, preserve_small_move_weights_threshold: float = 0.0) -> dict:
    fbx_export_scale=float(manifest.get('fbx_export_scale') or 1.0)
    if abs(fbx_export_scale) < 1e-8:
        fbx_export_scale=1.0
    accum=collections.defaultdict(lambda: {'pos':[0.0,0.0,0.0], 'norm':[0.0,0.0,0.0], 'uv':[0.0,0.0], 'uv_samples':[], 'n':0, 'weights':collections.defaultdict(float), 'wn':0})
    verts=fbx['vertices']
    cp_seq=fbx['polygon_cp_sequence']
    skipped_misaligned = 0
    for pv_i,(sm_i,src_i) in enumerate(cp_map):
        if allowed_submeshes is not None and int(sm_i) not in allowed_submeshes:
            continue
        fbx_pv_i = pv_index_map[pv_i] if pv_index_map is not None and pv_i < len(pv_index_map) else pv_i
        if fbx_pv_i >= len(cp_seq):
            raise ValueError(f'FBX polygon vertex stream ended early at {fbx_pv_i}; expected at least {len(cp_map)}')
        cp_i=cp_seq[fbx_pv_i]
        if cp_i < 0 or cp_i >= len(verts):
            raise ValueError(f'invalid FBX polygon index {cp_i}')
        key=(sm_i,src_i); a=accum[key]
        p=verts[cp_i]
        native_pos = _native_stored_pos(shape, manifest, sm_i, src_i)
        if native_pos is not None:
            scaled_p = (float(p[0]) / fbx_export_scale, float(p[1]) / fbx_export_scale, float(p[2]) / fbx_export_scale)
            dist = math.dist(native_pos, scaled_p)
            if dist > 250.0:
                skipped_misaligned += 1
                continue
        _add_pos_to_accum(a, p)
        if fbx['normal_by_pv'] and fbx_pv_i < len(fbx['normal_by_pv']): n=fbx['normal_by_pv'][fbx_pv_i]
        elif fbx['normal_by_cp'] and cp_i < len(fbx['normal_by_cp']): n=fbx['normal_by_cp'][cp_i]
        else: n=None
        if n:
            a['norm'][0]+=n[0]; a['norm'][1]+=n[1]; a['norm'][2]+=n[2]
        if fbx['uv_by_pv'] and fbx_pv_i < len(fbx['uv_by_pv']): uv=fbx['uv_by_pv'][fbx_pv_i]
        elif fbx['uv_by_cp'] and cp_i < len(fbx['uv_by_cp']): uv=fbx['uv_by_cp'][cp_i]
        else: uv=None
        if uv:
            a['uv_samples'].append((uv[0], uv[1]))
            _add_uv_to_accum(a, uv)
        wmap=fbx['weights_by_cp'].get(cp_i)
        if wmap:
            for bn,wt in wmap.items(): a['weights'][bn]+=wt
            a['wn'] += 1
        a['n'] += 1
    bones=manifest.get('bones',[])
    bone_name_to_id={b['name']:int(b['idx']) for b in bones}
    for b in bones:
        bone_name_to_id.setdefault(str(b['name']).replace(' ','_'), int(b['idx']))
    patched=0; weights_patched=0; reduced=0; uv_conflicts_preserved=0; position_conflicts_preserved=0; weights_preserved_for_small_moves=0
    submeshes=manifest['mesh_stats']
    for (sm_i,src_i),a in accum.items():
        n=max(1,a['n'])
        sm=submeshes[sm_i]
        stride=parse_int_maybe_hex(sm['vertex_stride'])
        off=parse_int_maybe_hex(sm['vertex_start']) + src_i*stride
        layout={64:'skin64',76:'blend76',52:'blend52',60:'blend60',48:'skin48',40:'skin40'}.get(stride,'skin64')
        native_pos = _native_stored_pos(shape, manifest, sm_i, src_i)
        pos, pos_conflict = _pos_from_accum(a, n, fbx_export_scale, native_pos)
        if pos_conflict:
            position_conflicts_preserved += 1
        preserve_weights = False
        if preserve_small_move_weights_threshold > 0.0 and native_pos is not None:
            preserve_weights = math.dist(native_pos, pos) <= preserve_small_move_weights_threshold
        norm=norm3((a['norm'][0]/n,a['norm'][1]/n,a['norm'][2]/n)) if any(abs(x)>1e-8 for x in a['norm']) else None
        stored_native_uv = _native_stored_uv(shape, manifest, sm_i, src_i)
        native_uv = (stored_native_uv[0], 1.0 - stored_native_uv[1]) if stored_native_uv is not None else None
        uv, uv_conflict = _uv_from_accum(a, n, layout, native_uv)
        if uv_conflict:
            uv_conflicts_preserved += 1
        struct.pack_into('>3f', shape, off, *pos)
        if layout == 'skin64':
            if uv: struct.pack_into('>2f', shape, off+20, float(uv[0]), float(1.0-uv[1]))
            if norm: struct.pack_into('>3f', shape, off+28, *norm)
            if a['wn'] and preserve_weights:
                weights_preserved_for_small_moves += 1
            elif a['wn']:
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
            if a['wn'] and preserve_weights:
                weights_preserved_for_small_moves += 1
            elif a['wn']:
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
            if a['wn'] and preserve_weights:
                weights_preserved_for_small_moves += 1
            elif a['wn']:
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
        else:
            if uv: struct.pack_into('>2f', shape, off+32, float(uv[0]), float(1.0-uv[1]))
            if norm: struct.pack_into('>3f', shape, off+40, *norm)
            if a['wn'] and preserve_weights:
                weights_preserved_for_small_moves += 1
            elif a['wn']:
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
    return {'source_vertices_patched':patched,'weights_patched':weights_patched,'weights_preserved_for_small_moves':weights_preserved_for_small_moves,'weights_reduced_to_source_limits':reduced,'uv_conflicts_preserved':uv_conflicts_preserved,'position_conflicts_preserved':position_conflicts_preserved,'misaligned_polygon_corners_skipped':skipped_misaligned,'fbx_export_scale':fbx_export_scale}

def patch_mesh_added_topology_editor_style(shape: bytearray, fbx: dict, manifest: dict, report: dict, cp_map: list[tuple[int, int]] | None = None) -> bool:
    old_cp_count = int(manifest.get('control_points') or 0)
    old_face_count = int(manifest.get('triangles') or 0)
    faces = [list(int(v) for v in f) for f in (fbx.get('polygon_faces') or []) if len(f) == 3]
    vertices = list(fbx.get('vertices') or [])
    if old_cp_count <= 0 or old_face_count <= 0 or len(faces) <= old_face_count:
        return False

    pristine_shape = bytes(shape)
    added_face_count = len(faces) - old_face_count
    if added_face_count <= 0:
        return False
    existing_patch = {}

    old_shape = pristine_shape
    try:
        import bdg_to_fbx_extract_all as bdg
        import topology_bdg_writer as topo
        parser, entries = _bundle_entries_from_bytes(pristine_shape)
        mesh_entries = [e for e in entries if e.get('file_type') == 17 and e.get('is_resource')]
        main_entries = [e for e in entries if e.get('file_type') == 17 and not e.get('is_resource')]
        if len(mesh_entries) != 1 or len(main_entries) != 1:
            return False
        submeshes, _skipped = bdg.choose_meshes(pristine_shape, int(manifest.get('bone_count') or 0))
        endian = '>' if parser.is_big_endian else '<'
        topo._find_mesh_descriptors(pristine_shape, main_entries[0], mesh_entries[0], submeshes, endian)
    except Exception as exc:
        report['mesh_patch'] = {'status': f'error: topology setup failed: {exc}'}
        return True

    cp_to_pv = {}
    for pv_i, cp_i in enumerate(fbx.get('polygon_cp_sequence') or []):
        cp_to_pv.setdefault(int(cp_i), pv_i)

    def normal_for_cp(cp_i: int):
        pv_i = cp_to_pv.get(cp_i)
        if pv_i is not None and fbx.get('normal_by_pv') and pv_i < len(fbx['normal_by_pv']):
            return fbx['normal_by_pv'][pv_i]
        if fbx.get('normal_by_cp') and cp_i < len(fbx['normal_by_cp']):
            return fbx['normal_by_cp'][cp_i]
        return (0.0, 0.0, 1.0)

    def uv_for_cp(cp_i: int):
        pv_i = cp_to_pv.get(cp_i)
        uv = None
        if pv_i is not None and fbx.get('uv_by_pv') and pv_i < len(fbx['uv_by_pv']):
            uv = fbx['uv_by_pv'][pv_i]
        elif fbx.get('uv_by_cp') and cp_i < len(fbx['uv_by_cp']):
            uv = fbx['uv_by_cp'][cp_i]
        if uv is None:
            return (0.0, 0.0)
        return (float(uv[0]), float(1.0 - uv[1]))

    fbx_export_scale = float(manifest.get('fbx_export_scale') or 1.0)
    if abs(fbx_export_scale) < 1e-8:
        fbx_export_scale = 1.0
    payload_vertices = [
        (float(p[0]) / fbx_export_scale, float(p[1]) / fbx_export_scale, float(p[2]) / fbx_export_scale)
        for p in vertices
    ]
    payload_normals = [tuple(map(float, normal_for_cp(i))) for i in range(len(vertices))]
    payload_uvs = [uv_for_cp(i) for i in range(len(vertices))]
    bones = manifest.get('bones', [])
    bone_name_to_id = {b['name']: int(b['idx']) for b in bones}
    for b in bones:
        bone_name_to_id.setdefault(str(b['name']).replace(' ', '_'), int(b['idx']))
    payload_weights = [
        choose_top_weights(fbx.get('weights_by_cp', {}).get(i, {}), bone_name_to_id, 4)
        for i in range(len(vertices))
    ]

    vertex_src = []
    tri_groups = []
    source_items = []
    cp_i = 0
    for sm_i, sm in enumerate(submeshes):
        display_records = []
        try:
            for cmd in topo._parse_dl_commands_with_records(
                old_shape,
                int(sm['dl_start']),
                int(sm.get('dl_end', sm['v_start'])),
                int(sm.get('index_width', 6)),
            ):
                for rec_face in cmd.get('faces') or []:
                    for rec in rec_face:
                        display_records.append({
                            'raw_hex': bytes(rec.get('raw') or b'').hex(),
                            'a': int(rec.get('a', 0)),
                            'b': int(rec.get('b', rec.get('a', 0))),
                            'c': int(rec.get('c', rec.get('a', 0))),
                            'd': int(rec.get('d', rec.get('a', 0))),
                        })
        except Exception:
            display_records = []
        corner = 0
        layout = str(sm['layout'])
        stride = int(sm['v_stride'])
        for face in sm.get('faces') or []:
            for src_idx in face:
                rec = display_records[corner] if corner < len(display_records) else None
                src = (int(sm['v_start']), int(src_idx), layout, rec) if rec else (int(sm['v_start']), int(src_idx), layout)
                vertex_src.append(src)
                native_pos=(0.0,0.0,0.0); native_uv=(0.0,0.0); native_normal=(0.0,0.0,1.0)
                try:
                    native_pos,native_uv,native_normal,_native_weights=bdg.parse_vertex_by_layout(
                        pristine_shape,
                        int(sm['v_start'])+int(src_idx)*stride,
                        int(manifest.get('bone_count') or 0),
                        layout,
                    )
                except Exception:
                    pass
                source_items.append({
                    'cp': cp_i,
                    'sm_i': sm_i,
                    'src': src,
                    'pos': tuple(map(float,native_pos)),
                    'uv': tuple(map(float,native_uv)),
                    'normal': tuple(map(float,native_normal)),
                })
                cp_i += 1
                corner += 1
            tri_groups.append(sm_i)
    if cp_map is not None and len(cp_map) == old_cp_count and len(vertex_src) != old_cp_count:
        full_cp=[(int(item['sm_i']),int(item['src'][1])) for item in source_items]
        drop=_duplicate_seam_face_drops(pristine_shape,manifest,full_cp)
        if drop:
            native_face_count=len(source_items)//3
            source_items=[
                dict(item)
                for face_i in range(native_face_count) if face_i not in drop
                for item in source_items[face_i*3:face_i*3+3]
            ]
            for target_cp,item in enumerate(source_items):
                item['cp']=target_cp
            vertex_src=[item['src'] for item in source_items]
            tri_groups=[int(source_items[face_i*3]['sm_i']) for face_i in range(old_face_count)]
    if len(vertex_src) != old_cp_count or len(tri_groups) != old_face_count:
        report['mesh_patch']={
            'status':'error: topology source map could not align with exported duplicate-seam filtering',
            'native_polygon_vertices':len(vertex_src),
            'expected_polygon_vertices':old_cp_count,
            'native_faces':len(tri_groups),
            'expected_faces':old_face_count,
            'filtered_cp_map_vertices':len(cp_map) if cp_map is not None else None,
        }
        return True

    boundaries = [0]
    for i in range(1, len(tri_groups)):
        if tri_groups[i] != tri_groups[i - 1]:
            boundaries.append(i)
    boundaries.append(old_face_count)

    def face_insert_score(insert_face: int):
        tail_faces = old_face_count - insert_face
        if tail_faces <= 0:
            return (float('inf'), 0)
        sample_step = max(1, tail_faces // 256)
        total = 0.0
        count = 0
        for old_face_i in range(insert_face, old_face_count, sample_step):
            new_face_i = old_face_i + added_face_count
            if new_face_i >= len(faces):
                return (float('inf'), count)
            for corner_i, cp_i in enumerate(faces[new_face_i]):
                old_pv = old_face_i * 3 + corner_i
                if not (0 <= cp_i < len(payload_vertices)) or old_pv >= len(source_items):
                    continue
                old_pos = source_items[old_pv]['pos']
                new_pos = payload_vertices[cp_i]
                total += math.dist(old_pos, new_pos)
                count += 1
        return (total / max(1, count), count)

    inserted_by_new_control_points=[
        face_i for face_i,face in enumerate(faces)
        if any(int(cp) >= old_cp_count for cp in face)
    ]
    if (
        len(inserted_by_new_control_points) == added_face_count
        and inserted_by_new_control_points == list(range(inserted_by_new_control_points[0], inserted_by_new_control_points[0] + added_face_count))
    ):
        insert_face=inserted_by_new_control_points[0]
        best_score=0.0
    else:
        insert_face = old_face_count
        best_score = float('inf')
        for candidate in boundaries[:-1]:
            score, count = face_insert_score(candidate)
            if count and (score < best_score - 0.01 or (abs(score - best_score) <= 0.01 and candidate < insert_face)):
                best_score = score
                insert_face = candidate
        if best_score > 25.0:
            insert_face = old_face_count

    new_faces = faces[insert_face:insert_face + added_face_count]
    if not new_faces:
        return False
    if any(any(v >= len(vertices) for v in tri) for tri in new_faces):
        return False

    existing_pv_index_map = []
    added_cp_count = added_face_count * 3
    for pv_i in range(old_cp_count):
        face_i = pv_i // 3
        if face_i >= insert_face:
            existing_pv_index_map.append(pv_i + added_cp_count)
        else:
            existing_pv_index_map.append(pv_i)

    for face_i in range(insert_face + added_face_count, len(faces)):
        faces[face_i] = [int(v) - added_cp_count for v in faces[face_i]]

    if cp_map is not None:
        try:
            existing_patch = _patch_existing_mesh_vertices(bytearray(pristine_shape), fbx, manifest, cp_map, pv_index_map=existing_pv_index_map)
        except Exception as exc:
            report['mesh_patch'] = {'status': f'error: existing mesh patch before topology grow failed: {type(exc).__name__}: {exc}'}
            return True

    for cp_i in range(len(payload_vertices), old_cp_count):
        if cp_i < len(source_items):
            item = source_items[cp_i]
            payload_vertices.append(tuple(item.get('pos') or (0.0, 0.0, 0.0)))
            payload_normals.append(tuple(item.get('normal') or (0.0, 0.0, 1.0)))
            payload_uvs.append(tuple(item.get('uv') or (0.0, 0.0)))
            payload_weights.append([])
        else:
            payload_vertices.append((0.0, 0.0, 0.0))
            payload_normals.append((0.0, 0.0, 1.0))
            payload_uvs.append((0.0, 0.0))
            payload_weights.append([])

    def dist2(a, b):
        return ((float(a[0]) - float(b[0])) ** 2 +
                (float(a[1]) - float(b[1])) ** 2 +
                (float(a[2]) - float(b[2])) ** 2)

    def nearest(candidates, pos, uv=None, normal=None):
        best = None
        best_d = float('inf')
        for item in candidates:
            d = dist2(item['pos'], pos) * 0.001
            if uv is not None and item.get('uv') is not None:
                du = float(item['uv'][0]) - float(uv[0])
                dv = float(item['uv'][1]) - float(uv[1])
                d += (du * du + dv * dv) * 10000.0
            if normal is not None and item.get('normal') is not None:
                dn = (
                    (float(item['normal'][0]) - float(normal[0])) ** 2 +
                    (float(item['normal'][1]) - float(normal[1])) ** 2 +
                    (float(item['normal'][2]) - float(normal[2])) ** 2
                )
                d += dn
            if d < best_d:
                best = item
                best_d = d
        return best

    def duplicated_face_source_group(new_face):
        permutations=((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))
        new_pos=[payload_vertices[cp] for cp in new_face]
        new_uv=[payload_uvs[cp] if cp < len(payload_uvs) else (0.0,0.0) for cp in new_face]
        new_normal=[payload_normals[cp] if cp < len(payload_normals) else (0.0,0.0,1.0) for cp in new_face]
        new_center=tuple(sum(p[axis] for p in new_pos)/3.0 for axis in range(3))
        best=None
        for old_face_i in range(old_face_count):
            edited_face_i=old_face_i if old_face_i < insert_face else old_face_i + added_face_count
            if edited_face_i < 0 or edited_face_i >= len(faces):
                continue
            old_face=faces[edited_face_i]
            old_pos=[payload_vertices[cp] for cp in old_face]
            old_uv=[payload_uvs[cp] if cp < len(payload_uvs) else (0.0,0.0) for cp in old_face]
            old_normal=[payload_normals[cp] if cp < len(payload_normals) else (0.0,0.0,1.0) for cp in old_face]
            old_center=tuple(sum(p[axis] for p in old_pos)/3.0 for axis in range(3))
            for order in permutations:
                score=0.0
                for new_corner,old_corner in enumerate(order):
                    for axis in range(3):
                        delta=(new_pos[new_corner][axis]-new_center[axis])-(old_pos[old_corner][axis]-old_center[axis])
                        score += delta*delta
                        normal_delta=float(new_normal[new_corner][axis])-float(old_normal[old_corner][axis])
                        score += normal_delta*normal_delta
                    du=float(new_uv[new_corner][0])-float(old_uv[old_corner][0])
                    dv=float(new_uv[new_corner][1])-float(old_uv[old_corner][1])
                    score += (du*du+dv*dv)*10000.0
                if best is None or score < best[0]:
                    best=(score,int(tri_groups[old_face_i]),old_face_i)
        return best

    new_cp_ids = sorted({v for tri in new_faces for v in tri})
    group_votes = collections.Counter()
    duplicate_face_matches={}
    for face_offset,new_face in enumerate(new_faces):
        matched=duplicated_face_source_group(new_face)
        if matched is not None:
            duplicate_face_matches[face_offset]=matched
            group_votes[int(matched[1])] += len(new_face)
    nearest_by_new_cp = {}
    if not group_votes:
        for cp in new_cp_ids:
            item = nearest(source_items, payload_vertices[cp], payload_uvs[cp] if cp < len(payload_uvs) else None, payload_normals[cp] if cp < len(payload_normals) else None)
            if item:
                nearest_by_new_cp[cp] = item
                group_votes[int(item['sm_i'])] += 1
    if not group_votes:
        return False
    owner_group = group_votes.most_common(1)[0][0]
    source_items_by_group = collections.defaultdict(list)
    for item in source_items:
        source_items_by_group[int(item['sm_i'])].append(item)
    layout_face_counts = collections.Counter()
    for gi in tri_groups:
        try:
            layout_face_counts[(str(submeshes[int(gi)]['layout']), int(gi))] += 1
        except Exception:
            pass
    primary_group_by_layout = {}
    for (layout, gi), count in layout_face_counts.items():
        old = primary_group_by_layout.get(layout)
        if old is None or count > layout_face_counts[(layout, old)]:
            primary_group_by_layout[layout] = gi

    new_face_groups = []
    raw_new_face_groups = []
    for face_i in range(insert_face, insert_face + added_face_count):
        face_offset=face_i-insert_face
        matched=duplicate_face_matches.get(face_offset)
        if matched is not None:
            raw_group=int(matched[1])
        else:
            face_votes = collections.Counter()
            for cp in faces[face_i]:
                item = nearest_by_new_cp.get(cp)
                if item is None:
                    item = nearest(source_items, payload_vertices[cp], payload_uvs[cp] if cp < len(payload_uvs) else None, payload_normals[cp] if cp < len(payload_normals) else None)
                    if item is not None:
                        nearest_by_new_cp[cp] = item
                if item is not None:
                    face_votes[int(item['sm_i'])] += 1
            raw_group = face_votes.most_common(1)[0][0] if face_votes else owner_group
        raw_new_face_groups.append(raw_group)
        try:
            raw_layout = str(submeshes[int(raw_group)]['layout'])
            face_group = primary_group_by_layout.get(raw_layout, raw_group)
        except Exception:
            face_group = raw_group
        new_face_groups.append(face_group)

    def native_uv_for_source(sm_i: int, src_i: int):
        return _native_stored_uv(pristine_shape, manifest, sm_i, src_i)

    split_uv_vertices = 0
    split_uv_faces = set()
    preserve_original_face_keys = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    can_split_old_face_uvs = len(vertices) >= old_cp_count

    def corner_uv_for_face(face_i: int, corner_i: int, cp_i: int):
        pv_i = face_i * 3 + corner_i
        uv = None
        if fbx.get('uv_by_pv') and pv_i < len(fbx['uv_by_pv']):
            uv = fbx['uv_by_pv'][pv_i]
        elif fbx.get('uv_by_cp') and cp_i < len(fbx['uv_by_cp']):
            uv = fbx['uv_by_cp'][cp_i]
        if uv is None:
            return None
        return (float(uv[0]), float(1.0 - uv[1]))

    def corner_normal_for_face(face_i: int, corner_i: int, cp_i: int):
        pv_i = face_i * 3 + corner_i
        if fbx.get('normal_by_pv') and pv_i < len(fbx['normal_by_pv']):
            return tuple(map(float, fbx['normal_by_pv'][pv_i]))
        if fbx.get('normal_by_cp') and cp_i < len(fbx['normal_by_cp']):
            return tuple(map(float, fbx['normal_by_cp'][cp_i]))
        return payload_normals[cp_i] if cp_i < len(payload_normals) else (0.0, 0.0, 1.0)

    def corner_pos_for_face(face_i: int, corner_i: int, cp_i: int):
        pv_i = face_i * 3 + corner_i
        src_cp = None
        if fbx.get('polygon_cp_sequence') and pv_i < len(fbx['polygon_cp_sequence']):
            src_cp = int(fbx['polygon_cp_sequence'][pv_i])
        elif cp_i < len(vertices):
            src_cp = cp_i
        if src_cp is not None and 0 <= src_cp < len(vertices):
            p = vertices[src_cp]
            return (
                float(p[0]) / fbx_export_scale,
                float(p[1]) / fbx_export_scale,
                float(p[2]) / fbx_export_scale,
            )
        return payload_vertices[cp_i] if cp_i < len(payload_vertices) else (0.0, 0.0, 0.0)

    def preserve_original_face(face_i: int):
        try:
            original_key = []
            for original_cp in faces[face_i]:
                real_cp = original_cp
                if not (0 <= real_cp < len(cp_map)):
                    src = vertex_src[real_cp] if 0 <= real_cp < len(vertex_src) else None
                    if isinstance(src, tuple) and len(src) >= 4 and src[2] == 'virtual':
                        tmpl = src[3]
                        _sm, _src = int(tmpl[0]), int(tmpl[1])
                    else:
                        original_key = []
                        break
                else:
                    _sm, _src = cp_map[real_cp]
                if int(_sm) != owner_group:
                    original_key = []
                    break
                original_key.append(int(_src))
            if len(original_key) == 3 and len(set(original_key)) == 3:
                preserve_original_face_keys[int(owner_group)].append(tuple(sorted(original_key)))
        except Exception:
            pass

    if can_split_old_face_uvs:
        for face_i in range(old_face_count):
            for corner_i, cp_i in enumerate(faces[face_i]):
                if not (0 <= cp_i < len(vertex_src)):
                    continue
                src = vertex_src[cp_i]
                try:
                    sm_i, src_i = cp_map[cp_i]
                    layout = str(src[2])
                except Exception:
                    continue
                if sm_i != owner_group:
                    continue
                uv = corner_uv_for_face(face_i, corner_i, cp_i)
                if uv is None:
                    continue
                by_source[(sm_i, src_i, layout)].append((face_i, corner_i, cp_i, uv))

    for src, uses in by_source.items():
        if len(uses) < 2:
            continue
        us = [float(u[3][0]) for u in uses]
        vs = [float(u[3][1]) for u in uses]
        uv_spread = max(max(us) - min(us), max(vs) - min(vs))
        if uv_spread <= 1e-5:
            continue
        groups = collections.defaultdict(list)
        for use in uses:
            uv = use[3]
            groups[(round(float(uv[0]), 5), round(float(uv[1]), 5))].append(use)
        if len(groups) < 2:
            continue
        original_uv = native_uv_for_source(src[0], src[1])
        if original_uv is not None:
            keep_key = min(
                groups,
                key=lambda k: (float(k[0]) - float(original_uv[0])) ** 2 + (float(k[1]) - float(original_uv[1])) ** 2,
            )
        else:
            keep_key = max(groups, key=lambda k: len(groups[k]))
        for key, group_uses in groups.items():
            if key == keep_key:
                continue
            for face_i, corner_i, cp_i, _uv in group_uses:
                preserve_original_face(face_i)
                new_cp = len(payload_vertices)
                payload_vertices.append(payload_vertices[cp_i])
                payload_normals.append(corner_normal_for_face(face_i, corner_i, cp_i))
                payload_uvs.append(_uv)
                payload_weights.append(payload_weights[cp_i])
                vertex_src.append((-1, new_cp, 'virtual', tuple(vertex_src[cp_i])))
                faces[face_i][corner_i] = new_cp
                split_uv_vertices += 1
                split_uv_faces.add(face_i)

    split_position_vertices = 0
    split_position_faces = set()
    by_pos_source = collections.defaultdict(list)
    can_split_old_face_positions = existing_patch.get('position_conflicts_preserved', 0) >= 10
    if can_split_old_face_uvs and can_split_old_face_positions:
        for face_i in range(old_face_count):
            for corner_i, cp_i in enumerate(faces[face_i]):
                if not (0 <= cp_i < len(vertex_src)):
                    continue
                src = vertex_src[cp_i]
                try:
                    if isinstance(src, tuple) and len(src) >= 4 and src[2] == 'virtual':
                        tmpl = src[3]
                        sm_i, src_i, layout = int(tmpl[0]), int(tmpl[1]), str(tmpl[2])
                    else:
                        sm_i, src_i = cp_map[cp_i]
                        layout = str(src[2])
                except Exception:
                    continue
                if sm_i != owner_group:
                    continue
                pos = corner_pos_for_face(face_i, corner_i, cp_i)
                by_pos_source[(sm_i, src_i, layout)].append((face_i, corner_i, cp_i, pos))

    for src, uses in by_pos_source.items():
        if len(uses) < 2:
            continue
        xs = [float(u[3][0]) for u in uses]
        ys = [float(u[3][1]) for u in uses]
        zs = [float(u[3][2]) for u in uses]
        pos_spread = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        if pos_spread <= 2.0:
            continue
        groups = collections.defaultdict(list)
        for use in uses:
            pos = use[3]
            groups[(round(float(pos[0]), 5), round(float(pos[1]), 5), round(float(pos[2]), 5))].append(use)
        if len(groups) < 2:
            continue
        keep_key = max(groups, key=lambda k: (len(groups[k]), k))
        for key, group_uses in groups.items():
            if key == keep_key:
                continue
            for face_i, corner_i, cp_i, pos in group_uses:
                preserve_original_face(face_i)
                current_cp = faces[face_i][corner_i]
                new_cp = len(payload_vertices)
                payload_vertices.append(pos)
                payload_normals.append(corner_normal_for_face(face_i, corner_i, current_cp))
                _uv = corner_uv_for_face(face_i, corner_i, current_cp)
                if _uv is None and current_cp < len(payload_uvs):
                    _uv = payload_uvs[current_cp]
                payload_uvs.append(_uv if _uv is not None else (0.0, 0.0))
                payload_weights.append(payload_weights[current_cp] if current_cp < len(payload_weights) else [])
                vertex_src.append((-1, new_cp, 'virtual', tuple(vertex_src[current_cp])))
                faces[face_i][corner_i] = new_cp
                split_position_vertices += 1
                split_position_faces.add(face_i)

    owner_candidates = [item for item in source_items if int(item['sm_i']) == owner_group] or source_items
    appended_virtual_vertices = 0
    for face_i in range(insert_face, insert_face + added_face_count):
        face_group = new_face_groups[face_i - insert_face] if face_i - insert_face < len(new_face_groups) else owner_group
        face_owner_candidates = source_items_by_group.get(int(face_group)) or owner_candidates
        for corner_i, cp_i in enumerate(faces[face_i]):
            pos = payload_vertices[cp_i]
            uv = corner_uv_for_face(face_i, corner_i, cp_i)
            if uv is None and cp_i < len(payload_uvs):
                uv = payload_uvs[cp_i]
            nrm = corner_normal_for_face(face_i, corner_i, cp_i)
            item = nearest(face_owner_candidates, pos, uv, nrm)
            if item is None:
                return False
            new_cp = len(payload_vertices)
            payload_vertices.append(pos)
            payload_normals.append(nrm)
            payload_uvs.append(uv if uv is not None else (0.0, 0.0))
            payload_weights.append(payload_weights[cp_i] if cp_i < len(payload_weights) else [])
            vertex_src.append((-1, new_cp, 'virtual', tuple(item['src'])))
            faces[face_i][corner_i] = new_cp
            appended_virtual_vertices += 1
    new_cp_ids = list(range(len(payload_vertices) - appended_virtual_vertices, len(payload_vertices)))
    for cp in range(old_cp_count, len(payload_vertices)):
        if cp < len(vertex_src):
            try:
                if str(vertex_src[cp][2]) == 'virtual':
                    continue
            except Exception:
                pass
        item = nearest(owner_candidates, payload_vertices[cp], payload_uvs[cp] if cp < len(payload_uvs) else None, payload_normals[cp] if cp < len(payload_normals) else None)
        if item is None:
            return False
        vertex_src.append((-1, cp, 'virtual', tuple(item['src'])))

    payload = {
        'vertices': payload_vertices,
        'normals': payload_normals,
        'uvs': payload_uvs,
        'weights': payload_weights,
        'triangles': faces,
        'tri_groups': tri_groups[:insert_face] + new_face_groups + tri_groups[insert_face:],
        'vertex_src': vertex_src,
        'preserve_original_display_lists': True,
        'preserve_original_face_keys': {str(k): v for k, v in preserve_original_face_keys.items()},
    }

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.BDG') as tmp:
            tmp.write(old_shape)
            tmp_name = tmp.name
        topo.save_topology_payload_to_bdg(tmp_name, payload)
        shape[:] = Path(tmp_name).read_bytes()
        try:
            live_submeshes, _live_skipped = bdg.choose_meshes(bytes(shape), int(manifest.get('bone_count') or 0))
            live_manifest = dict(manifest)
            live_stats = []
            original_stats = manifest.get('mesh_stats') or []
            for i, sm in enumerate(live_submeshes):
                item = dict(original_stats[i]) if i < len(original_stats) else {}
                item.update({
                    'submesh': i,
                    'layout': str(sm['layout']),
                    'display_list_start': hex(int(sm['dl_start'])),
                    'display_list_end': hex(int(sm.get('dl_end', sm['v_start']))),
                    'vertex_start': hex(int(sm['v_start'])),
                    'vertex_stride': int(sm['v_stride']),
                    'index_width': int(sm.get('index_width', 6)),
                })
                live_stats.append(item)
            live_manifest['mesh_stats'] = live_stats
            if cp_map is not None:
                existing_patch = _patch_existing_mesh_vertices(shape, fbx, live_manifest, cp_map, pv_index_map=existing_pv_index_map)
        except Exception as exc:
            existing_patch = dict(existing_patch)
            existing_patch['post_grow_patch_error'] = f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        report['mesh_patch'] = {
            'status': f'error: topology grow failed: {type(exc).__name__}: {exc}',
            'owner_template_submesh': owner_group,
            'new_faces': len(new_faces),
            'new_vertices': len(new_cp_ids),
        }
        return True
    finally:
        if tmp_name:
            try: Path(tmp_name).unlink()
            except Exception: pass

    report['mesh_patch'] = {
        'status': 'patched_added_topology',
        'new_faces': len(new_faces),
        'new_vertices': len(new_cp_ids),
        'appended_face_vertices': appended_virtual_vertices,
        'owner_template_submesh': owner_group,
        'template_group_votes': dict(group_votes),
        'raw_new_face_group_counts': dict(collections.Counter(raw_new_face_groups)),
        'new_face_group_counts': dict(collections.Counter(new_face_groups)),
        'insert_face': insert_face,
        'added_face_count': added_face_count,
        'insert_score': best_score,
        'existing_source_vertices_patched': existing_patch.get('source_vertices_patched', 0),
        'existing_weights_patched': existing_patch.get('weights_patched', 0),
        'existing_weights_preserved_for_small_moves': existing_patch.get('weights_preserved_for_small_moves', 0),
        'existing_weights_reduced_to_source_limits': existing_patch.get('weights_reduced_to_source_limits', 0),
        'existing_uv_conflicts_preserved': existing_patch.get('uv_conflicts_preserved', 0),
        'existing_position_conflicts_preserved': existing_patch.get('position_conflicts_preserved', 0),
        'misaligned_polygon_corners_skipped': existing_patch.get('misaligned_polygon_corners_skipped', 0),
        'split_uv_vertices': split_uv_vertices,
        'split_uv_faces': len(split_uv_faces),
        'split_position_vertices': split_position_vertices,
        'split_position_faces': len(split_position_faces),
        'new_weights_patched': sum(1 for cp in new_cp_ids if cp < len(payload_weights) and payload_weights[cp]),
        'old_size': len(old_shape),
        'new_size': len(shape),
        'fbx_export_scale': fbx_export_scale,
    }
    return True

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
        if patch_mesh_added_topology_editor_style(shape, fbx, manifest, report, cp_map):
            return
        report['mesh_patch']={'status':'skipped_topology_changed','fbx_polygon_vertices':len(cp_seq),'expected_polygon_vertices':len(cp_map)}; return
    fbx_export_scale=float(manifest.get('fbx_export_scale') or 1.0)
    if abs(fbx_export_scale) < 1e-8:
        fbx_export_scale=1.0
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
        _add_uv_to_accum(a, uv)
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
    patched=0; weights_patched=0; reduced=0; uv_conflicts_preserved=0
    submeshes=manifest['mesh_stats']
    for (sm_i,src_i),a in accum.items():
        n=max(1,a['n'])
        sm=submeshes[sm_i]
        stride=parse_int_maybe_hex(sm['vertex_stride'])
        off=parse_int_maybe_hex(sm['vertex_start']) + src_i*stride
        # Map stream stride to the matching packed vertex layout.
        layout={64:'skin64',76:'blend76',52:'blend52',60:'blend60',48:'skin48',40:'skin40'}.get(stride,'skin64')
        pos=(a['pos'][0]/n/fbx_export_scale,a['pos'][1]/n/fbx_export_scale,a['pos'][2]/n/fbx_export_scale)
        norm=norm3((a['norm'][0]/n,a['norm'][1]/n,a['norm'][2]/n)) if any(abs(x)>1e-8 for x in a['norm']) else None
        uv, uv_conflict = _uv_from_accum(a, n, layout)
        if uv_conflict:
            uv_conflicts_preserved += 1
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
    report['mesh_patch']={'status':'patched','source_vertices_patched':patched,'weights_patched':weights_patched,'weights_reduced_to_source_limits':reduced,'uv_conflicts_preserved':uv_conflicts_preserved,'fbx_version':fbx['version'],'fbx_export_scale':fbx_export_scale}

def patch_skeleton_from_fbx(shape: bytearray, extracted: Path, manifest: dict, report: dict, patch_unchanged=False):
    fbx_path=extracted / manifest.get('fbx','Godzilla2K.fbx')
    if not fbx_path.exists():
        report['skeleton_patch']={'status':'skipped_missing_fbx'}; return
    old=(manifest.get('file_hashes') or {}).get(manifest.get('fbx','Godzilla2K.fbx'))
    if old and sha256_file(fbx_path)==old and not patch_unchanged:
        report['skeleton_patch']={'status':'unchanged_not_patched'}; return
    fbx=extract_fbx_mesh(fbx_path)
    records=parse_skeleton_records(bytes(shape), manifest)
    # Flexible name map: exact, space->underscore, trailing clean names.
    models=fbx['bone_models']
    fbx_export_scale=float(manifest.get('fbx_export_scale') or 1.0)
    if abs(fbx_export_scale) < 1e-8:
        fbx_export_scale=1.0
    matched={}
    missing=[]
    for idx,r in records.items():
        cand=find_bdg_bone_model(models, r['name'])
        if cand:
            matched[idx]=cand
        else:
            missing.append(r['name'])
    children=collections.defaultdict(list)
    for idx,r in records.items():
        parent=int(r.get('parent', -1))
        if parent >= 0:
            children[parent].append(idx)
    locked_indices={idx for idx,r in records.items() if not matched.get(idx)}
    # Blender reparents surviving children when an EditBone is deleted. Preserve
    # their native rest records so that reparenting is not mistaken for a move,
    # but only lock animation tracks for bones actually absent from the FBX.
    preserved_indices=set(locked_indices)
    stack=list(locked_indices)
    while stack:
        parent_idx=stack.pop()
        for child_idx in children.get(parent_idx, []):
            if child_idx not in preserved_indices:
                preserved_indices.add(child_idx)
                stack.append(child_idx)
    patched=0; unchanged=[]; rotations_ignored=0
    changed_bones=[]
    normalized_baseline=manifest.get('fbx_bone_translation_baseline') or {}
    translation_epsilon=1e-3
    for idx,r in records.items():
        if idx in preserved_indices:
            continue
        name=r['name']
        cand=matched.get(idx)
        if not cand:
            continue
        off=r['off']
        if cand.get('rotation_euler_xyz_deg'):
            rotations_ignored += 1
        if cand.get('translation'):
            tx,ty,tz=cand['translation']
            old_t=tuple(float(v) for v in r['t'])
            baseline_t=find_bdg_bone_model(normalized_baseline, name)
            if isinstance(baseline_t, (list, tuple)) and len(baseline_t) >= 3:
                new_t=tuple(
                    old_t[i] + ((tx,ty,tz)[i] - float(baseline_t[i])) / fbx_export_scale
                    for i in range(3)
                )
            else:
                new_t=(tx/fbx_export_scale, ty/fbx_export_scale, tz/fbx_export_scale)
            if any(abs(new_t[i]-old_t[i]) > translation_epsilon for i in range(3)):
                struct.pack_into('>3f', shape, off+32, *new_t)
                patched += 1
                changed_bones.append({
                    'idx': idx,
                    'name': name,
                    'old_local_translation': old_t,
                    'new_local_translation': new_t,
                })
            else:
                unchanged.append(name)
    report['skeleton_patch']={
        'status':'patched' if patched else 'skipped_no_changed_bone_positions',
        'position_bones_patched':patched,
        'changed_bones':changed_bones,
        'deleted_or_missing_bones_preserved':missing[:20],
        'deleted_or_missing_count':len(missing),
        'deleted_or_missing_bone_indices':sorted(int(i) for i in locked_indices),
        'deleted_descendant_bones_preserved':len(preserved_indices)-len(locked_indices),
        'unchanged_bones_seen':len(unchanged),
        'rotation_values_seen_but_preserved':rotations_ignored,
        'fbx_export_scale':fbx_export_scale,
        'lock_rule':'delete a bone node from the edited FBX to preserve that BDG skeleton record',
        'writeback_scope':'local translation only; animations and rest rotations are preserved',
    }

def patch_type4_skeleton_pose(data: bytearray, report: dict, stream_name: str):
    skel=report.get('skeleton_patch') or {}
    changed=skel.get('changed_bones') or []
    if not changed:
        report.setdefault('type4_skeleton_patches', []).append({'stream':stream_name,'status':'skipped_no_changed_bones'})
        return
    try:
        _parser, entries=_bundle_entries_from_bytes(bytes(data))
    except Exception as e:
        report.setdefault('type4_skeleton_patches', []).append({'stream':stream_name,'status':f'skipped_parse_error: {type(e).__name__}: {e}'})
        return

    changed_by_idx={int(b['idx']):b for b in changed if 'idx' in b}
    patches=[]
    for e in entries:
        name=str(e.get('name') or '')
        lname=name.lower()
        if e.get('is_resource') or int(e.get('file_type', -1)) != 4:
            continue
        if 'skeleton' not in lname or 'intro_cam' in lname or 'camera' in lname:
            continue
        base=int(e['offset']); size=int(e['size'])
        if base < 0 or size < 0 or base + size > len(data) or size < 0x38:
            continue
        try:
            count=struct.unpack_from('>I', data, base+0x2c)[0]
        except Exception:
            continue
        if count <= 0 or count > 512 or 0x38 + count*4 > size:
            continue
        patched=[]
        for bone_idx,bone in changed_by_idx.items():
            if bone_idx < 0 or bone_idx >= count:
                continue
            try:
                rec_rel=struct.unpack_from('>I', data, base+0x38+bone_idx*4)[0]
                rec=base+rec_rel
                if rec < base or rec + 0x24 > base + size:
                    continue
                rec_idx=struct.unpack_from('>i', data, rec)[0]
                trans_rel=struct.unpack_from('>I', data, rec+0x1c)[0]
                trans=rec + trans_rel
                if rec_idx != bone_idx or trans < base or trans + 12 > base + size:
                    continue
                old_t=tuple(float(v) for v in bone.get('old_local_translation', ()))
                new_t=tuple(float(v) for v in bone.get('new_local_translation', ()))
                if len(old_t) != 3 or len(new_t) != 3:
                    continue
                cur=struct.unpack_from('>3f', data, trans)
                if any(abs(float(cur[i])-old_t[i]) > 0.05 for i in range(3)):
                    patched.append({
                        'idx':bone_idx,
                        'name':bone.get('name',''),
                        'status':'skipped_current_translation_mismatch',
                        'offset':hex(trans-base),
                        'current':cur,
                        'expected_old':old_t,
                    })
                    continue
                struct.pack_into('>3f', data, trans, *new_t)
                patched.append({
                    'idx':bone_idx,
                    'name':bone.get('name',''),
                    'status':'patched',
                    'offset':hex(trans-base),
                    'old_local_translation':old_t,
                    'new_local_translation':new_t,
                })
            except Exception as ex:
                patched.append({'idx':bone_idx,'name':bone.get('name',''),'status':f'error: {type(ex).__name__}: {ex}'})
        if patched:
            patches.append({'stream':stream_name,'resource':name,'resource_offset':hex(base),'bone_count':count,'bones':patched})
    if patches:
        report.setdefault('type4_skeleton_patches', []).extend(patches)
    else:
        report.setdefault('type4_skeleton_patches', []).append({'stream':stream_name,'status':'skipped_no_matching_type4_skeleton'})

def _walk_type3_skeleton_records(data: bytes | bytearray, base: int, size: int, strings: list[str]):
    records={}
    end=base+size
    root=base+0x40
    def rec(off):
        if off < base or off + 48 > end:
            raise ValueError(f'Type 3 skeleton record is outside its resource: {off-base:#x}')
        idx,parent,nchild,name_idx=struct.unpack_from('>4i', data, off)
        if idx < 0 or idx > 4096 or parent < -1 or parent > 4096 or nchild < 0 or nchild > 512:
            raise ValueError(f'Invalid Type 3 skeleton record at {off-base:#x}')
        if off + 48 + nchild*4 > end:
            raise ValueError(f'Type 3 child table is outside its resource at {off-base:#x}')
        q=struct.unpack_from('>4f', data, off+16)
        t=struct.unpack_from('>3f', data, off+32)
        child_rels=struct.unpack_from('>'+('I'*nchild), data, off+48) if nchild else ()
        name=strings[name_idx] if 0 <= name_idx < len(strings) else ''
        return {'idx':idx,'parent':parent,'nchild':nchild,'name_idx':name_idx,'name':name,'q':q,'t':t,'off':off,'children':[base+c for c in child_rels]}
    def walk(off):
        if off in {r['off'] for r in records.values()}:
            return
        r=rec(off)
        if r['idx'] in records:
            return
        records[r['idx']]=r
        for c in r['children']:
            if base <= c and c+48 <= end:
                walk(c)
    walk(root)
    return records

def patch_type3_skeleton_from_report(data: bytearray, report: dict, stream_name: str):
    skel=report.get('skeleton_patch') or {}
    changed=skel.get('changed_bones') or []
    if not changed:
        report.setdefault('type3_skeleton_patches', []).append({'stream':stream_name,'status':'skipped_no_changed_bones'})
        return
    try:
        parser, entries=_bundle_entries_from_bytes(bytes(data))
        strings=parse_shapes_string_table(bytes(data), int(parser.string_offset))
    except Exception as e:
        report.setdefault('type3_skeleton_patches', []).append({'stream':stream_name,'status':f'skipped_parse_error: {type(e).__name__}: {e}'})
        return
    changed_by_idx={int(b['idx']):b for b in changed if 'idx' in b}
    patches=[]
    for e in entries:
        name=str(e.get('name') or '')
        lname=name.lower()
        if e.get('is_resource') or int(e.get('file_type', -1)) != 3:
            continue
        if 'skeleton' not in lname or 'intro_cam' in lname or 'camera' in lname:
            continue
        base=int(e['offset']); size=int(e['size'])
        if base < 0 or size < 0 or base + size > len(data) or size < 0x80:
            continue
        try:
            records=_walk_type3_skeleton_records(data, base, size, strings)
        except Exception:
            continue
        patched=[]
        for bone_idx,bone in changed_by_idx.items():
            r=records.get(bone_idx)
            if not r:
                continue
            old_t=tuple(float(v) for v in bone.get('old_local_translation', ()))
            new_t=tuple(float(v) for v in bone.get('new_local_translation', ()))
            if len(old_t) != 3 or len(new_t) != 3:
                continue
            cur=tuple(float(v) for v in r['t'])
            if all(abs(cur[i]-new_t[i]) <= 0.0005 for i in range(3)):
                patched.append({'idx':bone_idx,'name':bone.get('name',''),'status':'already_patched','offset':hex(r['off']-base)})
                continue
            if any(abs(cur[i]-old_t[i]) > 0.05 for i in range(3)):
                patched.append({
                    'idx':bone_idx,
                    'name':bone.get('name',''),
                    'status':'skipped_current_translation_mismatch',
                    'offset':hex(r['off']-base),
                    'current':cur,
                    'expected_old':old_t,
                })
                continue
            struct.pack_into('>3f', data, r['off']+32, *new_t)
            patched.append({
                'idx':bone_idx,
                'name':bone.get('name',''),
                'status':'patched',
                'offset':hex(r['off']-base),
                'old_local_translation':old_t,
                'new_local_translation':new_t,
            })
        if patched:
            patches.append({'stream':stream_name,'resource':name,'resource_offset':hex(base),'bones':patched})
    if patches:
        report.setdefault('type3_skeleton_patches', []).extend(patches)
    else:
        report.setdefault('type3_skeleton_patches', []).append({'stream':stream_name,'status':'skipped_no_matching_type3_skeleton'})

def raw_anims_changed(extracted: Path, manifest: dict, filename: str) -> bool:
    hashes=manifest.get('file_hashes') or {}
    rel=f'animations_raw/{filename}'
    p=extracted/rel
    if not p.exists(): return False
    old=hashes.get(rel)
    if not old: return True
    return sha256_file(p) != old

def quat_xyz_to_i16(q):
    return tuple(max(-32767, min(32767, int(round(float(v) * 32767.0)))) for v in q[:3])

def patch_native_track_rotation(payload: bytearray, resource_abs: int, track: dict, xyz: tuple[int, int, int]) -> int:
    rel=parse_int_maybe_hex(track.get('track_rel', 0))
    count=int(track.get('record_count') or 0)
    layout=str(track.get('layout') or '')
    changed=0
    if layout == 'explicit_qxyz_time':
        for i in range(count):
            off=resource_abs + rel + 4 + i*8
            if off + 6 > len(payload):
                break
            struct.pack_into('>hhh', payload, off, *xyz)
            changed += 1
    elif layout == 'continuation_time_qxyz':
        for i in range(count):
            off=resource_abs + rel + i*8 + 2
            if off + 6 > len(payload):
                break
            struct.pack_into('>hhh', payload, off, *xyz)
            changed += 1
    elif layout == 'first_qxyz_then_time_qxyz':
        first=resource_abs + rel + 4
        if first + 6 <= len(payload):
            struct.pack_into('>hhh', payload, first, *xyz)
            changed += 1
        for i in range(1,count):
            off=resource_abs + rel + 10 + (i-1)*8 + 2
            if off + 6 > len(payload):
                break
            struct.pack_into('>hhh', payload, off, *xyz)
            changed += 1
    return changed

def patch_deleted_bone_animation_locks(anim: bytearray, extracted: Path, manifest: dict, report: dict):
    skeleton_report=report.get('skeleton_patch') or {}
    locked={int(i) for i in skeleton_report.get('deleted_or_missing_bone_indices') or []}
    if not locked:
        report['animation_resource_patches'].append({'status':'skipped_no_deleted_bone_locks'})
        return
    if not anim:
        report['animation_resource_patches'].append({'status':'skipped_deleted_bone_locks_no_animation_bdg','locked_bones':sorted(locked)})
        return
    tracks_by_rid={}
    for res in manifest.get('animation_resource_locations', []):
        tracks=res.get('native_rotation_tracks')
        if tracks is not None:
            rid=int(res.get('resource_id', -1))
            tracks_by_rid[rid]={
                'resource_id':rid,
                'name':res.get('name'),
                'tracks':tracks,
            }

    # Legacy exports stored this metadata in animations_raw. Continue accepting
    # those projects, but new exports keep the required layout data in the log.
    native_path=extracted/'animations_raw'/'animation_native_tracks_v11.json'
    raw_native=[]
    if not tracks_by_rid and native_path.exists():
        try:
            raw_native=json.loads(native_path.read_text(encoding='utf-8'))
        except Exception:
            raw_native=[]
    if not tracks_by_rid and not raw_native:
        report['animation_resource_patches'].append({'status':'skipped_deleted_bone_locks_no_native_track_manifest','locked_bones':sorted(locked)})
        return
    for clip in raw_native:
        rid=int(clip.get('resource_id', -1))
        tracks_by_rid[rid]=clip
    bones={int(b['idx']):b for b in manifest.get('bones', []) if 'idx' in b}
    total_tracks=0; total_records=0; clips=[]
    for res in manifest.get('animation_resource_locations', []):
        rid=int(res.get('resource_id', -1))
        native=tracks_by_rid.get(rid)
        if not native:
            continue
        resource_abs=parse_int_maybe_hex(res.get('absolute_offset', 0))
        clip_report={'clip':res.get('name') or native.get('name'), 'resource_id':rid, 'tracks':[]}
        for tr in native.get('tracks', []):
            bone_id=int(tr.get('bone_id', -1))
            if bone_id not in locked or bone_id not in bones:
                continue
            xyz=quat_xyz_to_i16(bones[bone_id].get('native_local_quaternion_xyzw', bones[bone_id].get('local_quaternion_xyzw', (0,0,0,1))))
            changed=patch_native_track_rotation(anim, resource_abs, tr, xyz)
            if changed:
                total_tracks += 1
                total_records += changed
                clip_report['tracks'].append({'bone_id':bone_id,'bone_name':tr.get('bone_name'),'records_patched':changed,'rest_xyz_i16':list(xyz)})
        if clip_report['tracks']:
            clip_report['status']='patched_deleted_bone_locks'
            clips.append(clip_report)
    report['animation_resource_patches'].append({
        'status':'patched_deleted_bone_locks' if total_tracks else 'skipped_no_matching_deleted_bone_tracks',
        'locked_bones':sorted(locked),
        'tracks_patched':total_tracks,
        'records_patched':total_records,
        'clips':clips[:20],
    })

def patch_raw_anims(anim: bytearray, extracted: Path, manifest: dict, report: dict, patch_unchanged=False):
    raw_dir=extracted/'animations_raw'
    if not raw_dir.exists():
        report['animation_resource_patches'].append({'status':'skipped_no_animations_raw_folder'}); return
    for res in manifest.get('animation_resource_locations', []):
        filename=res.get('safe_filename') or f"{res['name']}.bin"
        raw=raw_dir/filename
        if not raw.exists():
            report['animation_resource_patches'].append({'clip':res['name'],'status':'skipped_missing_raw_bin'}); continue
        if not patch_unchanged and not raw_anims_changed(extracted,manifest,filename):
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
    ap.add_argument('--no-action-anims', action='store_true', help='Do not import edited FBX Actions into Type 4 animations')
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

    shape_name=manifest.get('source') or manifest.get('source_shapes') or ''
    anim_name=manifest.get('animation_source') or manifest.get('source_anim') or ''
    shape_src=find_case_insensitive(root, shape_name)
    anim_src=find_case_insensitive(root, anim_name)
    if shape_src is None or anim_src is None:
        raise SystemExit(f'Missing original BDGs beside Import.bat: {shape_name}, {anim_name}')
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
            'Changed BDG skeleton local positions import from FBX bone nodes; deleted/missing bone nodes preserve the original skeleton record.',
            'Edited FBX Actions are rebuilt as native Type 4 clips when the extraction contains an Action baseline.',
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
                if not args.no_action_anims and manifest.get('animation_action_baseline'):
                    from bdg_animation_import import import_bdg_actions
                    fbx_path=extracted / manifest.get('fbx','Godzilla2K.fbx')
                    anim, _action_results=import_bdg_actions(bytes(anim), fbx_path, manifest, report)
                    anim=bytearray(anim)
                elif not args.no_action_anims:
                    report['animation_action_import']=[{'status':'skipped_missing_action_baseline'}]
                patch_type3_skeleton_from_report(anim, report, 'animation')
                patch_type4_skeleton_pose(shape, report, 'shapes')
                patch_type4_skeleton_pose(anim, report, 'animation')
                patch_mesh_for_skeleton_position_edits(shape, manifest, report)
            except Exception as e:
                report['skeleton_patch']={'status':f'error: {type(e).__name__}: {e}'}
        if not args.no_raw_anims:
            patch_raw_anims(anim, extracted, manifest, report, patch_unchanged=args.patch_unchanged)
            patch_deleted_bone_animation_locks(anim, extracted, manifest, report)
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
