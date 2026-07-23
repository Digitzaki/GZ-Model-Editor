from pathlib import Path
import argparse, collections, hashlib, io, json, math, os, re, shutil, struct, sys, zipfile
from PIL import Image
TOOL_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOL_DIR.parent
for _p in (TOOL_DIR, ROOT_DIR):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
from decode_cmpr import decode_cmpr
from wii_tex_decode import decode_rgb565, decode_i8, decode_ia4, decode_ia8, decode_rgb5a3
from parser_core import PipeworksParser

CMD_QUADS=0x80; CMD_TRIS=0x90; CMD_TRI_STRIP=0x98; CMD_TRI_FAN=0xA0
VALID_DL={CMD_QUADS,CMD_TRIS,CMD_TRI_STRIP,CMD_TRI_FAN}
FBX_TICKS_PER_SECOND=46186158000
BDG_FBX_EXPORT_SCALE=10.0

ANIM_PREVIEW_MODE = os.environ.get('BDG_ANIM_PREVIEW_MODE', 'native_abs').strip().lower()
if ANIM_PREVIEW_MODE not in ('native_abs', 'root_stabilized', 'bind_delta'):
    ANIM_PREVIEW_MODE = 'native_abs'


def clean_arg(value):
    value=str(value).strip()
    while value and value[-1] in ('\"',"'"): value=value[:-1].rstrip()
    while value and value[0] in ('\"',"'"): value=value[1:].lstrip()
    return value or '.'

def be32(D,off): return struct.unpack('>I',D[off:off+4])[0]
def align(n,a=0x20): return (n+a-1)//a*a

def find_case(root,name):
    p=root/name
    if p.exists(): return p
    lname=name.lower()
    for q in root.iterdir():
        if q.name.lower()==lname: return q
    return None

def find_sets(root, all_mode=False):
    shapes=sorted([p for p in root.iterdir() if p.is_file() and p.name.lower().endswith('_shapes.bdg')])
    if not shapes: raise SystemExit('No *_Shapes.BDG found beside Extract.bat.')
    if len(shapes)>1 and not all_mode:
        raise SystemExit('More than one *_Shapes.BDG found. Run Extract.bat v4/all mode or pass --all.')
    out=[]
    for shape in shapes:
        base=re.sub(r'_Shapes\.BDG$','',shape.name,flags=re.I)
        anim=find_case(root,base+'.BDG')
        pvms=sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower()=='.pvm' and (p.name.lower().startswith(base.lower()) or (base=='Mechagodzilla_2' and p.name.lower().startswith('mechagodzilla2')))])
        if not anim:
            print(f'[mesh-only] {base}: missing {base}.BDG; exporting model without animation clips')
        out.append((base,shape,anim,pvms))
    return out

def find_strtab(D):
    # Header value at 0x38 is the real string table for normal BDGs; scan fallback handles odd files.
    starts=[]
    try:
        h=be32(D,0x38)
        if 0x100<=h<len(D): starts.append(h)
    except Exception: pass
    starts += list(range(0x100,0x3000,4))
    seen=set(); best=None
    for off in starts:
        if off in seen or off+8>len(D): continue
        seen.add(off)
        cnt=struct.unpack('<I',D[off:off+4])[0]
        if not (8 <= cnt <= 500): continue
        end=off+4+cnt*4
        if end>len(D): continue
        ptrs=struct.unpack('<'+'I'*cnt,D[off+4:end])
        if not ptrs or ptrs[0]<4 or max(ptrs)>0x10000: continue
        inc=sum(ptrs[i]<ptrs[i+1] for i in range(cnt-1))
        if inc<cnt*0.75: continue
        strings=[]; ok=0
        for p in ptrs:
            so=off+p
            if so>=len(D): break
            ze=D.find(b'\0',so,min(len(D),so+240))
            if ze<0: break
            raw=D[so:ze]
            if len(raw)<=180 and all(32<=b<127 for b in raw):
                strings.append(raw.decode('latin1')); ok+=1
            else: strings.append('')
        if ok>=cnt*0.75:
            score=ok + (200 if 'Bip01' in strings else 0) + (100 if any('SKELETON' in s.upper() for s in strings) else 0)
            if best is None or score>best[0]: best=(score,off,cnt,strings)
    if not best: raise ValueError('Could not locate BDG string table')
    return best[1], best[2], best[3]

def q_ok(q):
    n=sum(x*x for x in q)
    return all(math.isfinite(x) for x in q) and 0.45<n<1.55

def skel_rec(D,off):
    if off<0 or off+56>len(D): return None
    idx,parent,nchild,name_idx=struct.unpack('>4i',D[off:off+16])
    q=struct.unpack('>4f',D[off+16:off+32])
    t=struct.unpack('>3f',D[off+32:off+44])
    return idx,parent,nchild,name_idx,q,t

def find_skeleton(D,strings):
    if 'Bip01' not in strings: raise ValueError('No Bip01 in string table')
    bip=strings.index('Bip01'); cnt=len(strings)
    roots=[]
    for off in range(0x700,min(len(D)-64,0x60000),0x10):
        r=skel_rec(D,off)
        if not r: continue
        idx,parent,nchild,name_idx,q,t=r
        if idx==0 and parent==-1 and 0<nchild<32 and name_idx==bip and q_ok(q) and all(abs(v)<100000 for v in t):
            rels=struct.unpack('>'+('I'*nchild),D[off+48:off+48+4*nchild])
            roots.append((off,rels))
    for root,rels in roots:
        for base in range(max(0,root-0x2000),root+1,0x10):
            ok=True
            for rel in rels:
                r=skel_rec(D,base+rel)
                if not r: ok=False; break
                idx,parent,nchild,name_idx,q,t=r
                if not (0<idx<cnt and parent==0 and 0<=nchild<32 and 0<=name_idx<cnt and q_ok(q)): ok=False; break
            if not ok: continue
            records={}
            def walk(off):
                if off in records: return False
                r=skel_rec(D,off)
                if not r: return False
                idx,parent,nchild,name_idx,q,t=r
                if not (0<=idx<cnt and -1<=parent<cnt and 0<=nchild<32 and 0<=name_idx<cnt and q_ok(q)): return False
                records[off]={'idx':idx,'parent':parent,'nchild':nchild,'name_idx':name_idx,'name':strings[name_idx],'q':q,'t':t,'off':off,'children':[]}
                rels=struct.unpack('>'+('I'*nchild),D[off+48:off+48+4*nchild]) if nchild else ()
                for rel in rels:
                    child=base+rel; records[off]['children'].append(child)
                    if not walk(child): return False
                return True
            if walk(root) and len(records)>=10:
                byid={v['idx']:v for v in records.values()}
                if set(byid.keys())==set(range(max(byid.keys())+1)):
                    return base,root,byid
    raise ValueError('Could not decode skeleton records')

def qmat(q):
    x,y,z,w=q; n=x*x+y*y+z*z+w*w
    if n<1e-12: return [[1,0,0],[0,1,0],[0,0,1]]
    s=2/n; xx,yy,zz=x*x*s,y*y*s,z*z*s; xy,xz,yz=x*y*s,x*z*s,y*z*s; wx,wy,wz=w*x*s,w*y*s,w*z*s
    return [[1-yy-zz,xy-wz,xz+wy],[xy+wz,1-xx-zz,yz-wx],[xz-wy,yz+wx,1-xx-yy]]

def mm(a,b): return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def col_local(r):
    R=qmat(r['q']); x,y,z=r['t']
    return [[R[0][0],R[0][1],R[0][2],x],[R[1][0],R[1][1],R[1][2],y],[R[2][0],R[2][1],R[2][2],z],[0,0,0,1]]

def fbx_matrix_from_col(m):
    return [m[0][0],m[1][0],m[2][0],0, m[0][1],m[1][1],m[2][1],0, m[0][2],m[1][2],m[2][2],0, m[0][3],m[1][3],m[2][3],1]

def quat_to_euler_xyz_degrees(q):
    x,y,z,w=q
    sinr=2*(w*x+y*z); cosr=1-2*(x*x+y*y); roll=math.atan2(sinr,cosr)
    sinp=2*(w*y-z*x); pitch=math.copysign(math.pi/2,sinp) if abs(sinp)>=1 else math.asin(sinp)
    siny=2*(w*z+x*y); cosy=1-2*(y*y+z*z); yaw=math.atan2(siny,cosy)
    return (math.degrees(roll),math.degrees(pitch),math.degrees(yaw))

def norm3(v):
    l=math.sqrt(sum(x*x for x in v)) or 1.0
    return tuple(x/l for x in v)

def read_display_list_width(D,start,index_width=6):
    pos=start; faces=[]; used=[]; cmds=0
    # Skip NOP padding after the first real primitive.
    while pos<len(D)-3:
        if D[pos] == 0x00 and cmds > 0:
            pos += 1
            continue
        if D[pos] not in VALID_DL:
            break
        op=D[pos]; count=int.from_bytes(D[pos+1:pos+3],'big')
        if count<3 or count>4096 or pos+3+index_width*count>len(D): break
        pos+=3; verts=[]
        for _ in range(count):
            if index_width==6:
                a,b,c=struct.unpack('>3H',D[pos:pos+6]); pos+=6
            elif index_width==3:
                a,b,c=D[pos],D[pos+1],D[pos+2]; pos+=3
            elif index_width==4:
                a,b,c=D[pos],D[pos+1],D[pos+2]; pos+=4
            elif index_width==8:
                a,b,c,d=struct.unpack('>4H',D[pos:pos+8]); pos+=8
            else:
                raise ValueError('unsupported display-list index width')
            verts.append(a); used.append((a,b,c))
        cmds+=1
        if op==CMD_QUADS:
            for i in range(0,len(verts)-3,4): faces += [(verts[i],verts[i+1],verts[i+2]),(verts[i],verts[i+2],verts[i+3])]
        elif op==CMD_TRIS:
            for i in range(0,len(verts)-2,3): faces.append((verts[i],verts[i+1],verts[i+2]))
        elif op==CMD_TRI_STRIP:
            for i in range(len(verts)-2):
                a,b,c=verts[i],verts[i+1],verts[i+2]
                if a!=b and b!=c and a!=c: faces.append((a,b,c) if i%2==0 else (b,a,c))
        elif op==CMD_TRI_FAN and len(verts)>=3:
            root=verts[0]
            for i in range(1,len(verts)-1):
                a,b,c=root,verts[i],verts[i+1]
                if a!=b and b!=c and a!=c: faces.append((a,b,c))
    return faces,used,pos,cmds

def read_display_list(D,start):
    return read_display_list_width(D,start,6)

def find_display_lists(D):
    # Supports normal 16-bit and compact 8-bit display-list streams.
    out=[]
    for s in range(0x1000,len(D)-100,0x20):
        for index_width in (6,8,4,3):
            faces,used,end,cmds=read_display_list_width(D,s,index_width)
            if cmds>=2 and len(used)>=12 and len(faces)>=4:
                ident=sum(1 for a,b,c in used if a==b==c)/len(used)
                if ident>0.90 and (not out or s>=out[-1]['dl_end']):
                    out.append({'dl_start':s,'dl_end':end,'used':used,'faces':faces,'cmds':cmds,'min_index':min(a for a,b,c in used),'max_index':max(a for a,b,c in used),'ident':ident,'index_width':index_width})
                    break
    return out

def valid_float(f): return math.isfinite(f) and abs(f)<100000

def norm_ok(v): return all(math.isfinite(x) and abs(x)<10 for x in v) and 0.001<sum(x*x for x in v)<20

def validate_layout(D,vstart,count,layout,bone_count):
    stride={'skin64':64,'blend76':76,'skin40':40,'skin48':48,'blend52':52,'blend60':60}[layout]
    if vstart+count*stride>len(D): return -1
    idxs=list(range(min(count,20))) + ([int(i*(count-1)/49) for i in range(50)] if count>50 else list(range(count)))
    ok=0; total=0
    # Reject layouts whose normal/UV offsets land on the wrong fields.
    n_mags=[]; uvs_seen=[]
    for idx in sorted(set(idxs)):
        off=vstart+idx*stride
        try:
            if layout=='skin64':
                x,y,z,w=struct.unpack('>4f',D[off:off+16]); b0,b1=struct.unpack('>2H',D[off+16:off+20]); u,v=struct.unpack('>2f',D[off+20:off+28]); n=struct.unpack('>3f',D[off+28:off+40])
                cond=all(valid_float(f) for f in [x,y,z,w,u,v]) and -0.05<=w<=1.05 and b0<bone_count and b1<bone_count and norm_ok(n) and -50<=u<=50 and -50<=v<=50
            elif layout=='blend76':
                x,y,z=struct.unpack('>3f',D[off:off+12]); w0,w1,w2=struct.unpack('>3f',D[off+12:off+24]); slots=struct.unpack('>4H',D[off+24:off+32]); u,v=struct.unpack('>2f',D[off+32:off+40]); n=struct.unpack('>3f',D[off+40:off+52])
                ws=[w0,w1,w2,1-(w0+w1+w2)]
                cond=all(valid_float(f) for f in [x,y,z,u,v,w0,w1,w2]) and all(-0.05<=wt<=1.05 for wt in ws) and all(s<bone_count for s in slots) and norm_ok(n) and -50<=u<=50 and -50<=v<=50
            elif layout in ('blend52','blend60'):
                # Compact blended stream; blend60 has trailing pad.
                x,y,z=struct.unpack('>3f',D[off:off+12]); w0,w1,w2=struct.unpack('>3f',D[off+12:off+24]); slots=struct.unpack('>4H',D[off+24:off+32]); n=struct.unpack('>3f',D[off+32:off+44]); u,v=struct.unpack('>2f',D[off+44:off+52])
                ws=[w0,w1,w2,1-(w0+w1+w2)]
                cond=all(valid_float(f) for f in [x,y,z,u,v,w0,w1,w2]) and all(-0.05<=wt<=1.05 for wt in ws) and all(s<bone_count for s in slots) and norm_ok(n) and -50<=u<=50 and -50<=v<=50
            elif layout=='skin48':
                # skin40 plus 8 bytes pad.
                x,y,z,w=struct.unpack('>4f',D[off:off+16]); b0,b1=struct.unpack('>2H',D[off+16:off+20]); n=struct.unpack('>3f',D[off+20:off+32]); u,v=struct.unpack('>2f',D[off+32:off+40])
                cond=all(valid_float(f) for f in [x,y,z,w,u,v]) and -0.05<=w<=1.05 and b0<bone_count and b1<bone_count and norm_ok(n) and -50<=u<=50 and -50<=v<=50
            else: # skin40 compact two-influence stream
                x,y,z,w=struct.unpack('>4f',D[off:off+16]); b0,b1=struct.unpack('>2H',D[off+16:off+20]); n=struct.unpack('>3f',D[off+20:off+32]); u,v=struct.unpack('>2f',D[off+32:off+40])
                cond=all(valid_float(f) for f in [x,y,z,w,u,v]) and -0.05<=w<=1.05 and b0<bone_count and b1<bone_count and norm_ok(n) and -50<=u<=50 and -50<=v<=50
        except Exception:
            cond=False; n=None; u=v=None
        total+=1
        if cond:
            ok+=1
            try:
                n_mags.append(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
                uvs_seen.append((u,v))
            except Exception:
                pass
    base=ok/max(total,1)
    if base<0.5 or not n_mags:
        return base
    tight=sum(1 for m in n_mags if 0.7<=m<=1.3)/len(n_mags)
    if uvs_seen:
        us={round(u,4) for u,_ in uvs_seen}; vs={round(v,4) for _,v in uvs_seen}
        varies=1.0 if (len(us)>=max(2,len(uvs_seen)//8) and len(vs)>=max(2,len(uvs_seen)//8)) else 0.4
    else:
        varies=0.0
    return base*tight*varies

def descriptor_meshes(D,bone_count):
    """Use type-17 mesh descriptors when present.

    The byte scanner below is intentionally conservative and rejects display
    lists where many GX records do not use identical position/normal/uv indices.
    Added topology can be valid while reusing original material/shader attribute
    indices, so descriptor-owned streams are the better source for edited files.
    """
    try:
        from parser_core import PipeworksParser
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.BDG') as f:
            f.write(D)
            tmp=f.name
        try:
            parser=PipeworksParser(tmp)
            entries=parser.parse()
        finally:
            try: os.unlink(tmp)
            except Exception: pass
    except Exception:
        return []
    mesh_res=[e for e in entries if e.get('file_type')==17 and e.get('is_resource')]
    mesh_main=[e for e in entries if e.get('file_type')==17 and not e.get('is_resource')]
    if len(mesh_res)!=1 or len(mesh_main)!=1:
        return []
    res=mesh_res[0]; main=mesh_main[0]
    res_start=int(res['offset']); res_size=int(res['size'])
    B=D[int(main['offset']):int(main['offset'])+int(main['size'])]
    stride_layout={64:'skin64',76:'blend76',52:'blend52',60:'blend60',48:'skin48',40:'skin40'}
    out=[]; seen=set()
    for base in range(0,max(0,len(B)-56),4):
        try:
            record_count=be32(B,base+0)
            marker=be32(B,base+4)
            rel_dl=be32(B,base+8)
            dl_size=be32(B,base+12)
            attr_count=be32(B,base+16)
            v_count=be32(B,base+32)
            stride=be32(B,base+36)
            flags=be32(B,base+40)
            rel_v=be32(B,base+48)
            v_size=be32(B,base+52)
        except Exception:
            continue
        if attr_count!=3 or stride not in stride_layout:
            continue
        if v_count<=0 or v_size!=v_count*stride:
            continue
        if not (0<=rel_dl<rel_v<=res_size and rel_v+v_size<=res_size and rel_dl+dl_size==rel_v):
            continue
        if (rel_dl,rel_v) in seen:
            continue
        dl_start=res_start+rel_dl
        v_start=res_start+rel_v
        widths=(3,4,6,8) if marker==0x8 else (6,8,3,4)
        parsed=None
        for iw in widths:
            try:
                faces,used,end,cmds=read_display_list_width(D,dl_start,iw)
            except Exception:
                continue
            if end!=v_start or cmds<=0 or not used:
                continue
            if max(a for a,b,c in used)>=v_count:
                continue
            parsed=(iw,faces,used,cmds)
            break
        if parsed is None:
            continue
        layout=stride_layout[stride]
        score=validate_layout(D,v_start,v_count,layout,bone_count)
        if score<0.5:
            continue
        iw,faces,used,cmds=parsed
        seen.add((rel_dl,rel_v))
        out.append({
            'dl_start':dl_start,'dl_end':v_start,'used':used,'faces':faces,'cmds':cmds,
            'min_index':min(a for a,b,c in used),'max_index':max(a for a,b,c in used),
            'ident':sum(1 for a,b,c in used if a==b==c)/len(used),
            'index_width':iw,'v_start':v_start,'v_count':v_count,'v_stride':stride,
            'layout':layout,'validation_score':score,
            'descriptor_base':int(main['offset'])+base,'descriptor_flags':flags,
            'descriptor_record_count':record_count,
        })
    out.sort(key=lambda sm: sm['dl_start'])
    return out

def choose_meshes(D,bone_count):
    desc_sub=descriptor_meshes(D,bone_count)
    if desc_sub:
        skipped=[]
        desc_ranges={(sm['dl_start'],sm['dl_end']) for sm in desc_sub}
        for dl in find_display_lists(D):
            if (dl['dl_start'],dl['dl_end']) not in desc_ranges:
                skipped.append({'display_list_start':hex(dl['dl_start']),'display_list_end':hex(dl['dl_end']),'vertex_count_from_indices':dl['max_index']+1,'best_validation_score':'descriptor_not_used','best_layout':None})
        return desc_sub,skipped
    sub=[]; skipped=[]
    for dl in find_display_lists(D):
        count=dl['max_index']+1; base_vs=align(dl['dl_end'])
        best=(-1,None,None)
        # search small pad because several streams are aligned after local metadata
        for vstart in range(base_vs,min(base_vs+0x4000,len(D)),0x20):
            for layout in ('skin64','blend76','blend52','blend60','skin48','skin40'):
                sc=validate_layout(D,vstart,count,layout,bone_count)
                if sc>best[0]: best=(sc,vstart,layout)
        if best[0]>=0.85:
            sub.append({**dl,'v_start':best[1],'v_count':count,'v_stride':{'skin64':64,'blend76':76,'blend52':52,'blend60':60,'skin48':48,'skin40':40}[best[2]],'layout':best[2],'validation_score':best[0],'index_width':dl.get('index_width',6)})
        else:
            skipped.append({'display_list_start':hex(dl['dl_start']),'display_list_end':hex(dl['dl_end']),'vertex_count_from_indices':count,'best_validation_score':best[0],'best_layout':best[2]})
    if not sub: raise ValueError('No supported mesh display-list/vertex-stream pair found')
    return sub,skipped

def parse_skin64(D,off,bone_count):
    x,y,z,w=struct.unpack('>4f',D[off:off+16]); b0,b1=struct.unpack('>2H',D[off+16:off+20]); u,v=struct.unpack('>2f',D[off+20:off+28]); n=norm3(struct.unpack('>3f',D[off+28:off+40]))
    if b0>=bone_count or b1>=bone_count: raise ValueError('bone id out of range')
    if b0==b1: pairs=[(b0,1.0)]
    else: pairs=[(b0,max(0,min(1,w))),(b1,max(0,min(1,1-w)))]
    s=sum(wt for _,wt in pairs) or 1.0
    return (x,y,z),(u,1-v),n,[(b,wt/s) for b,wt in pairs if wt>1e-6]


def parse_skin40(D,off,bone_count):
    x,y,z,w=struct.unpack('>4f',D[off:off+16]); b0,b1=struct.unpack('>2H',D[off+16:off+20]); n=norm3(struct.unpack('>3f',D[off+20:off+32])); u,v=struct.unpack('>2f',D[off+32:off+40])
    if b0>=bone_count or b1>=bone_count: raise ValueError('bone id out of range')
    if b0==b1: pairs=[(b0,1.0)]
    else: pairs=[(b0,max(0,min(1,w))),(b1,max(0,min(1,1-w)))]
    s=sum(wt for _,wt in pairs) or 1.0
    return (x,y,z),(u,1-v),n,[(b,wt/s) for b,wt in pairs if wt>1e-6]

def parse_blend76(D,off,bone_count):
    x,y,z=struct.unpack('>3f',D[off:off+12]); w0,w1,w2=struct.unpack('>3f',D[off+12:off+24]); slots=struct.unpack('>4H',D[off+24:off+32]); u,v=struct.unpack('>2f',D[off+32:off+40]); n=norm3(struct.unpack('>3f',D[off+40:off+52]))
    ws=[w0,w1,w2,1-(w0+w1+w2)]; acc=collections.defaultdict(float)
    for b,wt in zip(slots,ws):
        if b>=bone_count: raise ValueError('bone id out of range')
        if wt>1e-6: acc[b]+=max(0,wt)
    s=sum(acc.values()) or 1.0
    return (x,y,z),(u,1-v),n,[(b,wt/s) for b,wt in sorted(acc.items())]

def parse_blend52(D,off,bone_count):
    x,y,z=struct.unpack('>3f',D[off:off+12]); w0,w1,w2=struct.unpack('>3f',D[off+12:off+24]); slots=struct.unpack('>4H',D[off+24:off+32]); n=norm3(struct.unpack('>3f',D[off+32:off+44])); u,v=struct.unpack('>2f',D[off+44:off+52])
    ws=[w0,w1,w2,1-(w0+w1+w2)]; acc=collections.defaultdict(float)
    for b,wt in zip(slots,ws):
        if b>=bone_count: raise ValueError('bone id out of range')
        if wt>1e-6: acc[b]+=max(0,wt)
    s=sum(acc.values()) or 1.0
    return (x,y,z),(u,1-v),n,[(b,wt/s) for b,wt in sorted(acc.items())]

def parse_blend60(D,off,bone_count):
    return parse_blend52(D,off,bone_count)

def parse_skin48(D,off,bone_count):
    x,y,z,w=struct.unpack('>4f',D[off:off+16]); b0,b1=struct.unpack('>2H',D[off+16:off+20]); n=norm3(struct.unpack('>3f',D[off+20:off+32])); u,v=struct.unpack('>2f',D[off+32:off+40])
    if b0>=bone_count or b1>=bone_count: raise ValueError('bone id out of range')
    if b0==b1: pairs=[(b0,1.0)]
    else: pairs=[(b0,max(0,min(1,w))),(b1,max(0,min(1,1-w)))]
    s=sum(wt for _,wt in pairs) or 1.0
    return (x,y,z),(u,1-v),n,[(b,wt/s) for b,wt in pairs if wt>1e-6]

def parse_vertex_by_layout(D,off,bone_count,layout):
    if layout=='skin64': return parse_skin64(D,off,bone_count)
    if layout=='blend76': return parse_blend76(D,off,bone_count)
    if layout=='blend52': return parse_blend52(D,off,bone_count)
    if layout=='blend60': return parse_blend60(D,off,bone_count)
    if layout=='skin48': return parse_skin48(D,off,bone_count)
    return parse_skin40(D,off,bone_count)

def texture_entries(D):
    class _MemoryParser(PipeworksParser):
        def parse(self):
            self.file_data = D
            if self.file_data[0:9].decode('ascii', errors='ignore') != 'Pipeworks':
                raise ValueError('Not a Pipeworks bundle (header missing)')
            self.is_big_endian = struct.unpack('<H', self.file_data[0x2C:0x2E])[0] == 0
            self.string_offset = self.read_long(0x34)
            self.file_count = self.read_short(0x62)
            self.metadata_offset = self.read_long(0x64)
            self.main_data_offset = self.read_long(0x68)
            self.resource_data_offset = self.read_long(0x70)
            results = []
            for i in range(self.file_count):
                entry_offset = 0x78 + (i * 0x12)
                file_num = self.read_short(entry_offset)
                offset = self.read_long(entry_offset + 2)
                size = self.read_long(entry_offset + 6)
                res_offset = self.read_long(entry_offset + 10)
                res_size = self.read_long(entry_offset + 14)
                name, file_type = self.get_file_info(file_num)
                results.append({'file_num':file_num,'name':name,'file_type':file_type,'offset':offset+self.main_data_offset,'size':size,'is_resource':False,'toc_entry_offset':entry_offset})
                if res_size > 0:
                    results.append({'file_num':file_num,'name':f'{name}.resource','file_type':file_type,'offset':res_offset+self.resource_data_offset,'size':res_size,'is_resource':True,'toc_entry_offset':entry_offset})
            return results

    data_base=be32(D,0x70)
    entries=[]
    try:
        parsed=_MemoryParser('<memory>').parse()
        main_by_file = {int(e['file_num']): e for e in parsed if e.get('file_type') == 9 and not e.get('is_resource')}
        for e in parsed:
            if e.get('file_type') != 9 or not e.get('is_resource'):
                continue
            size=int(e['size'])
            name=str(e['name']).split('/',1)[-1].replace('.resource','')
            main = main_by_file.get(int(e['file_num']))
            fmt_code = None; w = None; h = None; mip_count = None
            if main:
                hdr = D[int(main['offset']):int(main['offset'])+16]
                if len(hdr) >= 16:
                    _unk, fmt_code = struct.unpack('>II', hdr[:8])
                    w, h, mip_count = struct.unpack('>HHH', hdr[8:14])
            if size>0 and int(e['offset'])+min(size,1)<=len(D):
                entries.append({'rid':int(e['file_num']),'rel':int(e['offset'])-data_base,'size':size,'abs':int(e['offset']),'type':9,'name':name,'fmt_code':fmt_code,'width':w,'height':h,'mip_count':mip_count})
        return entries,data_base
    except Exception:
        pass
    count=be32(D,0x60)
    # Fallback for odd bundles: Shapes.BDG resource-location table is 18 bytes from 0x80.
    for i in range(count):
        off=0x80+i*18
        if off+18>len(D): break
        typ=struct.unpack('>H',D[off:off+2])[0]
        rel,size=struct.unpack('>II',D[off+2:off+10])
        rid=struct.unpack('>H',D[off+10:off+12])[0]
        if size>0 and data_base+rel+min(size,1)<=len(D):
            if size in (0x2aac0,0x2aaa0,0xaaaa0,0x58200,0x15560,0x15600) or (typ in (0x68,0x42c) and size>0x10000):
                entries.append({'rid':rid,'rel':rel,'size':size,'abs':data_base+rel,'type':typ,'name':f'texture_rid_{rid}'})
    return entries,data_base

def decode_textures(D,strings,outdir,asset_name):
    texdir=outdir/'textures'; texdir.mkdir(parents=True,exist_ok=True)
    entries,data_base=texture_entries(D)
    specs=[]
    used=set()

    def named_entry(token):
        token=token.upper()
        for e in entries:
            if e['rid'] in used:
                continue
            if token in str(e.get('name','')).upper():
                used.add(e['rid'])
                return e
        return None

    def format_from_entry(e):
        code = e.get('fmt_code')
        w = int(e.get('width') or 0)
        h = int(e.get('height') or 0)
        if code == 0x02:
            return 'IA4', w, h
        if code == 0x03:
            return 'IA8', w, h
        if code == 0x04:
            return 'RGB565', w, h
        if code == 0x05:
            return 'RGB5A3', w, h
        if code == 0x0E:
            return 'CMPR', w, h
        size=int(e['size'])
        if size==0xAAAA0:
            return 'RGB565',512,512
        if size in (0x2AAC0,0x2AAA0):
            return 'CMPR',512,512
        if size in (0x15560,0x15600):
            return 'I8',256,256
        return None,0,0

    def add_named(token,suffix,prop=None):
        e=named_entry(token)
        if not e:
            return
        fmt,w,h = format_from_entry(e)
        if not fmt or not w or not h:
            return
        specs.append((suffix,fmt,e,w,h,prop))

    add_named('_512_C','C','DiffuseColor')
    add_named('_512_B','B',None)
    add_named('_512_S','S','SpecularColor')
    add_named('_256_M','M',None)

    if not specs:
        # Fallback order for bundles without texture names.
        cmpr=[e for e in entries if e['size']==0x2aac0]
        rgb=[e for e in entries if e['size']==0xaaaa0]
        i8=[e for e in entries if e['size'] in (0x15560,0x15600)]
        if cmpr: specs.append(('C','CMPR',cmpr[0],512,512,'DiffuseColor'))
        if rgb: specs.append(('B','RGB565',rgb[0],512,512,None))
        if len(cmpr)>1: specs.append(('S','CMPR',cmpr[1],512,512,'SpecularColor'))
        if i8: specs.append(('M','I8',i8[-1],256,256,None))
    tex_manifest=[]; bindings=[]
    for suffix,fmt,e,w,h,prop in specs:
        raw=D[e['abs']:]
        try:
            if fmt=='CMPR': img=decode_cmpr(raw[:w*h//2],w,h)
            elif fmt=='RGB565': img=decode_rgb565(raw[:w*h*2],w,h)
            elif fmt=='RGB5A3': img=decode_rgb5a3(raw[:w*h*2],w,h)
            elif fmt=='IA4': img=decode_ia4(raw[:w*h],w,h)
            elif fmt=='IA8': img=decode_ia8(raw[:w*h*2],w,h)
            else: img=decode_i8(raw[:w*h],w,h)
            fn=f'{asset_name}_{suffix}.png'; img.save(texdir/fn)
            tex_manifest.append({'file':fn,'name':e.get('name'),'format':fmt,'rid':e['rid'],'rel':hex(e['rel']),'size':hex(e['size']),'width':w,'height':h})
            if prop: bindings.append((asset_name+'_'+suffix,f'textures/{fn}',prop))
        except Exception as ex:
            tex_manifest.append({'suffix':suffix,'error':str(ex),'rid':e['rid'],'rel':hex(e['rel'])})
    # Blender normal map from raw B.
    raw_b=texdir/f'{asset_name}_B.png'
    if raw_b.exists():
        raw=Image.open(raw_b).convert('RGBA'); normal=Image.new('RGBA',raw.size); sp=raw.load(); dp=normal.load()
        for y in range(raw.height):
            for x in range(raw.width):
                r,g,b,a=sp[x,y]; nx=r/255*2-1; ny=g/255*2-1; nz=math.sqrt(max(0,1-min(1,nx*nx+ny*ny)))
                dp[x,y]=(int((nx*.5+.5)*255+.5),int((-ny*.5+.5)*255+.5),int((nz*.5+.5)*255+.5),a)
        nfn=f'{asset_name}_N.png'; normal.save(texdir/nfn); bindings.append((asset_name+'_N',f'textures/{nfn}','NormalMap'))
    return tex_manifest,bindings

# -------- FBX writer classes --------
class Prop:
    def __init__(self,code,value): self.code=code; self.value=value
class Arr:
    def __init__(self,code,values): self.code=code; self.values=list(values)
class Node:
    def __init__(self,name,props=None,children=None): self.name=name; self.props=props or []; self.children=children or []
def PInt(v): return Prop('I',int(v))
def PLong(v): return Prop('L',int(v))
def PDouble(v): return Prop('D',float(v))
def PBool(v): return Prop('C',bool(v))
def PStr(v): return Prop('S',str(v))
def PRaw(v): return Prop('R',bytes(v))
def ADouble(v): return Arr('d',v)
def AInt(v): return Arr('i',v)
def ALong(v): return Arr('l',v)
def AFloat(v): return Arr('f',v)
def pack_prop(p):
    if isinstance(p,Prop):
        c=p.code.encode(); v=p.value
        if p.code=='I': return c+struct.pack('<i',v)
        if p.code=='L': return c+struct.pack('<q',v)
        if p.code=='D': return c+struct.pack('<d',v)
        if p.code=='C': return c+(b'\x01' if v else b'\x00')
        if p.code=='S':
            b=v.encode('utf-8'); return c+struct.pack('<I',len(b))+b
        if p.code=='R': return c+struct.pack('<I',len(v))+v
    if isinstance(p,Arr):
        vals=p.values; c=p.code.encode()
        if p.code=='d': data=struct.pack('<%sd'%len(vals),*map(float,vals)) if vals else b''
        elif p.code=='f': data=struct.pack('<%sf'%len(vals),*map(float,vals)) if vals else b''
        elif p.code=='i': data=struct.pack('<%si'%len(vals),*map(int,vals)) if vals else b''
        elif p.code=='l': data=struct.pack('<%sq'%len(vals),*map(int,vals)) if vals else b''
        else: raise ValueError(p.code)
        return c+struct.pack('<III',len(vals),0,len(data))+data
    raise TypeError(type(p))
NULL_RECORD=b'\0'*13
def write_node(buf,node):
    start=buf.tell(); props=b''.join(pack_prop(p) for p in node.props); nb=node.name.encode('ascii')
    buf.write(b'\0'*12); buf.write(bytes([len(nb)])); buf.write(nb); buf.write(props)
    for ch in node.children: write_node(buf,ch)
    if node.children: buf.write(NULL_RECORD)
    end=buf.tell(); cur=end; buf.seek(start); buf.write(struct.pack('<III',end,len(node.props),len(props))); buf.write(bytes([len(nb)])); buf.write(nb); buf.seek(cur)
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

def anim_strtab(D):
    off=be32(D,0x34); cnt=struct.unpack('<I',D[off:off+4])[0]
    ptrs=struct.unpack('<'+'I'*cnt,D[off+4:off+4+4*cnt])
    strings=[]
    for p in ptrs:
        so=off+p; ze=D.find(b'\0',so)
        strings.append(D[so:ze].decode('latin1','replace'))
    return off,cnt,strings

def quat_candidates_from_stored_xyz(qx_i,qy_i,qz_i):
    qx=qx_i/32767.0; qy=qy_i/32767.0; qz=qz_i/32767.0; n2=qx*qx+qy*qy+qz*qz
    if n2>1.05: return None
    qw=math.sqrt(max(0,1-n2))
    out=[]
    for sw in (1.0,-1.0):
        q=(qx,qy,qz,sw*qw)
        ln=math.sqrt(sum(v*v for v in q)) or 1.0
        out.append(tuple(v/ln for v in q))
    return out

def quat_from_stored_xyz(qx_i,qy_i,qz_i):
    c=quat_candidates_from_stored_xyz(qx_i,qy_i,qz_i)
    return c[0] if c else None

def dot4(a,b):
    return sum(float(a[i])*float(b[i]) for i in range(4))

def solve_quat_w_sign(records, target=(0,0,0,1)):
    """BDG stores quaternion xyz and reconstructs w. Some clips need the
    negative-w branch for a bone/key. Choosing +w globally is a common reason
    wings, tails, and full-body roots rotate incorrectly. Solve the missing w
    sign per track by choosing the branch closest to the bind pose at the first
    key and continuous with the prior solved key after that."""
    solved=[]; prev=None
    target=q_normalize(target)
    for qx,qy,qz in records:
        c=quat_candidates_from_stored_xyz(qx,qy,qz)
        if not c: return None
        ref=prev if prev is not None else target
        # Pick the actual rotation branch closest to the reference rotation.
        q=max(c, key=lambda x: abs(dot4(x,ref)))
        # Keep quaternion signs continuous for FBX slerp.
        if prev is not None and dot4(q,prev)<0: q=tuple(-v for v in q)
        solved.append(q_normalize(q)); prev=solved[-1]
    return solved



def q_normalize(q):
    l=math.sqrt(sum(float(x)*float(x) for x in q)) or 1.0
    return tuple(float(x)/l for x in q)

def q_slerp(a,b,u):
    a=q_normalize(a); b=q_normalize(b)
    dot=sum(a[i]*b[i] for i in range(4))
    if dot < 0.0:
        b=tuple(-x for x in b); dot=-dot
    dot=max(-1.0,min(1.0,dot))
    if dot > 0.9995:
        return q_normalize(tuple(a[i] + u*(b[i]-a[i]) for i in range(4)))
    th0=math.acos(dot); sin0=math.sin(th0)
    th=th0*u; s0=math.cos(th)-dot*math.sin(th)/sin0; s1=math.sin(th)/sin0
    return q_normalize(tuple(s0*a[i]+s1*b[i] for i in range(4)))

def sample_quat_keys(times,quats,duration,fps=60.0):
    """Dense-sample BDG quaternion keys so FBX/Blender does not invent
    Euler component interpolation between sparse quaternion keys.
    """
    if not times or not quats: return [], []
    duration=max(float(duration or 0.0), max(times) if times else 0.0)
    if len(times)==1 or duration <= 1e-6:
        return [0.0, max(duration,1.0/fps)], [quats[0], quats[0]]
    pairs=sorted(zip(times,quats), key=lambda x:x[0])
    # Drop duplicate/non-monotonic keys after sorting; keep the first value.
    clean=[]
    for t,q in pairs:
        if not clean or t > clean[-1][0] + 1e-7:
            clean.append((float(t), q_normalize(q)))
    if len(clean)==1:
        return [0.0, max(duration,1.0/fps)], [clean[0][1], clean[0][1]]
    if clean[0][0] > 1e-7:
        clean.insert(0,(0.0,clean[0][1]))
    if clean[-1][0] < duration - 1e-7:
        clean.append((duration,clean[-1][1]))
    max_keys=900
    step=1.0/float(fps)
    n=min(max_keys, max(2, int(math.ceil(duration/step))+1))
    sample_times=[min(duration, i*duration/(n-1)) for i in range(n)]
    # Preserve exact source key times too.
    sample_times=sorted(set([round(t,8) for t in sample_times] + [round(t,8) for t,_ in clean]))
    out_q=[]; j=0
    for t in sample_times:
        while j+1 < len(clean) and clean[j+1][0] < t - 1e-7:
            j+=1
        if j+1 >= len(clean):
            out_q.append(clean[-1][1]); continue
        t0,q0=clean[j]; t1,q1=clean[j+1]
        if t <= t0 + 1e-7: out_q.append(q0)
        elif t1 <= t0 + 1e-7: out_q.append(q1)
        else: out_q.append(q_slerp(q0,q1,(t-t0)/(t1-t0)))
    return sample_times,out_q

def _triangle_uv_span(tri_uvs):
    if len(tri_uvs) != 3:
        return 0.0
    best = 0.0
    for a, b in ((0, 1), (1, 2), (2, 0)):
        du = float(tri_uvs[a][0]) - float(tri_uvs[b][0])
        dv = float(tri_uvs[a][1]) - float(tri_uvs[b][1])
        best = max(best, math.sqrt(du * du + dv * dv))
    return best

def _filter_duplicate_seam_faces(vertices, normals, uvs, vertex_weights):
    groups = collections.defaultdict(list)
    face_count = len(vertices) // 3
    for fi in range(face_count):
        start = fi * 3
        key = tuple(sorted(
            (round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4))
            for p in vertices[start:start + 3]
        ))
        groups[key].append((fi, _triangle_uv_span(uvs[start:start + 3])))

    far_duplicate_extra = 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        face_ids = [fi for fi, _span in rows]
        if max(face_ids) - min(face_ids) > 64:
            far_duplicate_extra += len(rows) - 1
    if far_duplicate_extra <= 0 or far_duplicate_extra > 64:
        poly_indices = []
        for fi in range(face_count):
            base = fi * 3
            poly_indices.extend([base, base + 1, -base - 3])
        return vertices, normals, uvs, vertex_weights, poly_indices, 0

    drop = set()
    for rows in groups.values():
        if len(rows) < 2:
            continue
        spans = [span for _fi, span in rows]
        face_ids = [fi for fi, _span in rows]
        if max(face_ids) - min(face_ids) <= 64:
            continue
        if max(spans) - min(spans) > 0.02:
            keep = min(rows, key=lambda row: row[1])[0]
        else:
            keep = max(rows, key=lambda row: row[0])[0]
        for fi, _span in rows:
            if fi != keep:
                drop.add(fi)

    if not drop:
        poly_indices = []
        for fi in range(face_count):
            base = fi * 3
            poly_indices.extend([base, base + 1, -base - 3])
        return vertices, normals, uvs, vertex_weights, poly_indices, 0

    out_vertices = []
    out_normals = []
    out_uvs = []
    out_weights = []
    out_poly = []
    for fi in range(face_count):
        if fi in drop:
            continue
        base = len(out_vertices)
        start = fi * 3
        out_vertices.extend(vertices[start:start + 3])
        out_normals.extend(normals[start:start + 3])
        out_uvs.extend(uvs[start:start + 3])
        out_weights.extend(vertex_weights[start:start + 3])
        out_poly.extend([base, base + 1, -base - 3])
    return out_vertices, out_normals, out_uvs, out_weights, out_poly, len(drop)



def q_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])

def q_mul(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)

def q_inv(q):
    q=q_normalize(q)
    return q_conjugate(q)

def stabilize_root_quats_to_bind(quats, bind_q):
    """Legacy helper kept for manifest compatibility. v19 no longer uses this
    for FBX Actions because Blender pose-bone animation must be exported as a
    rest-pose delta, not as an absolute native local quaternion.
    """
    if not quats:
        return quats
    first=q_normalize(quats[0])
    bind=q_normalize(bind_q)
    inv_first=q_inv(first)
    return [q_normalize(q_mul(bind, q_mul(inv_first, q_normalize(q)))) for q in quats]

def native_abs_to_blender_pose_delta_quat(abs_q, bind_q):
    """Convert a native BDG absolute local bone rotation to the value Blender
    expects on a pose bone Action. Blender pose-bone rotation channels are
    evaluated relative to the imported rest pose. Writing the absolute BDG local
    quaternion directly makes Blender effectively apply rest * absolute, which
    double-rotates already-posed bones like Anguirus' Bip01/root. The delta below
    satisfies: rest * delta == native_absolute.
    """
    return q_normalize(q_mul(q_inv(bind_q), q_normalize(abs_q)))

def unwrap_eulers(eulers):
    if not eulers: return []
    out=[list(eulers[0])]
    for e in eulers[1:]:
        prev=out[-1]; cur=list(e)
        for i in range(3):
            while cur[i]-prev[i] > 180.0: cur[i]-=360.0
            while cur[i]-prev[i] < -180.0: cur[i]+=360.0
        out.append(cur)
    return [tuple(x) for x in out]

def validate_times(times):
    # BDG animation tracks may be fully keyed, two-key, or static one-key tracks.
    # Older builds rejected count<3, which dropped constant pose bones and made
    # Actions evaluate those bones from bind pose instead of the game pose.
    if len(times) == 1:
        return 0 <= times[0] <= 65535
    if len(times) == 2:
        a,b = times
        return 0 <= a <= 65535 and 0 <= b <= 65535 and (b > a or (a >= 55000 and b <= 30000))
    if len(set(times))<2: return False
    drops=[i for i,(a,b) in enumerate(zip(times,times[1:])) if b+16<a]
    if len(drops)>1: return False
    if len(drops)==1:
        if drops[0]!=len(times)-2 or times[-2]<55000 or times[-1]>30000: return False
        return all(b>a for a,b in zip(times[:-2],times[1:-1]))
    return all(b>a for a,b in zip(times,times[1:]))

def sort_scale_keys(raw,duration):
    ordered=sorted(raw,key=lambda x:x[0])
    if any(b[0]<=a[0] for a,b in zip(ordered,ordered[1:])): return None
    dur=float(duration if duration>0 else 1)
    return [(t/65534.0*dur,q,rt) for t,q,rt in ordered]

def sort_scale_keys_allow_static(raw,duration):
    # Same as sort_scale_keys, but preserves one-key static tracks and accepts
    # two-key wrapped tracks. FBX needs at least one curve key; duplicate time
    # expansion happens later when writing curves.
    if len(raw) == 1:
        t,q,rt = raw[0]
        return [(float(t)/65534.0*float(duration if duration>0 else 1), q, rt)]
    return sort_scale_keys(raw,duration)

def decode_animations(anim_path,bone_count,bone_names,skeleton,asset_hint,outdir):
    if not anim_path or not anim_path.exists(): return [],[]
    D=anim_path.read_bytes(); obj_base=be32(D,0x68); res_count=be32(D,0x60)
    try: _,_,strings=anim_strtab(D)
    except Exception: return [],[]
    def res_name(rid):
        # Descriptor table is not needed for names; names occur in string table. Use descriptor when valid, else scan fallback.
        desc=be32(D,0x64)
        try:
            ni=struct.unpack('>I',D[desc+rid*16+4:desc+rid*16+8])[0]
            if 0<=ni<len(strings): return strings[ni]
        except Exception: pass
        return ''
    def res_loc(rid):
        off=0x76+rid*18
        rid2,rel,size=struct.unpack('>III',D[off:off+12])
        return obj_base+rel,size
    def is_anim_name(n):
        # Header validation is authoritative; names are only hints.
        u=n.upper()
        if not u or any(x in u for x in ['MESH','.PWK','.PRX','SHADER','TEXTURE']): return False
        return asset_hint.upper().replace('_','')[:5] in u.replace('_','') or any(k in u for k in ['BAD','IDLE','CREEP','WALK','RUN','8WAY','RUSH','ATTACK','ATK','ROAR','JUMP','THROW','GRAB','BLOCK','HIT','HURT','BEAM','BREATH','SPAWN','TAUNT','VICTORY','STUN','FALL','TURN','DODGE','WAKE','DEATH','BITE','KICK','TAIL','WING','WPN','INTRO','PARRY','BRACE','GRAPPLE','LAUNCH','PRONE','KNOCKDOWN'])
    def is_anim_header(off,size):
        if off+0x40>len(D) or off+size>len(D) or size<0x100:
            return False
        dur=struct.unpack('>f',D[off+0x1c:off+0x20])[0]
        sz=struct.unpack('>I',D[off+0x24:off+0x28])[0]
        rot_end=struct.unpack('>I',D[off+0x38:off+0x3c])[0]
        rot_start=struct.unpack('>I',D[off+0x3c:off+0x40])[0]
        if not (0.0 < dur < 120.0): return False
        if sz not in (0,size): return False
        if not (0x40 <= rot_start < rot_end < size): return False
        # BDG animation resources all use this small family of magic/header fields.
        if D[off:off+4] != b'\x00\x00\x00\x00': return False
        if D[off+4:off+8] != b'\x02\x00\x00\x00': return False
        return True
    def decode_explicit(base,rel,end,dur):
        if rel+4>end: return None
        hw,zero=struct.unpack('>HH',D[base+rel:base+rel+4]); bone=hw>>8; count=hw&0xff
        if zero!=0 or not(0<=bone<bone_count) or count<1 or count>180 or rel+4+count*8>end: return None
        times=[]; xyz=[]; raw_records=[]
        for k in range(count):
            qx,qy,qz,t=struct.unpack('>hhhH',D[base+rel+4+k*8:base+rel+12+k*8])
            times.append(t); xyz.append((qx,qy,qz)); raw_records.append((qx,qy,qz,t))
        if not validate_times(times): return None
        target=skeleton[bone]['q'] if bone in skeleton else (0,0,0,1)
        qs=solve_quat_w_sign(xyz,target)
        if qs is None: return None
        raw=[(t,q,rr) for t,q,rr in zip(times,qs,raw_records)]
        keys=sort_scale_keys_allow_static(raw,dur)
        if not keys: return None
        return {'bone':bone,'rel':rel,'end':rel+4+count*8,'count':count,'keys':keys,'raw_times':times,'layout':'explicit_qxyz_time'}
    def decode_cont(base,start,end,bone,dur):
        if start>=end or (end-start)%8 or not(0<=bone<bone_count): return None
        count=(end-start)//8
        if count<1 or count>180: return None
        times=[]; xyz=[]; raw_records=[]
        for rel in range(start,end,8):
            t,qx,qy,qz=struct.unpack('>Hhhh',D[base+rel:base+rel+8])
            times.append(t); xyz.append((qx,qy,qz)); raw_records.append((t,qx,qy,qz))
        if not validate_times(times): return None
        target=skeleton[bone]['q'] if bone in skeleton else (0,0,0,1)
        qs=solve_quat_w_sign(xyz,target)
        if qs is None: return None
        raw=[(t,q,rr) for t,q,rr in zip(times,qs,raw_records)]
        keys=sort_scale_keys_allow_static(raw,dur)
        if not keys: return None
        return {'bone':bone,'rel':start,'end':end,'count':count,'keys':keys,'raw_times':times,'layout':'continuation_time_qxyz'}
    def table_start(base,rot_end,rot_start=0x58):
        for rel in range(max(rot_start,rot_end-0x400),rot_end-8,2):
            ids=[]; p=rel
            while p+2<=rot_end:
                b=D[base+p]; z=D[base+p+1]
                if z!=0 or b>=bone_count: break
                if ids and b==0: break
                ids.append(b); p+=2
            if len(ids)>=4 and all(ids[i]+1==ids[i+1] for i in range(len(ids)-1)): return rel
        return None
    decoded=[]; raw_entries=[]
    rawdir=outdir/'animations_raw'; rawdir.mkdir(parents=True, exist_ok=True)
    for rid in range(3,res_count):
        name=res_name(rid)
        try:
            off,size=res_loc(rid)
            if not is_anim_header(off,size): continue
            raw_descriptor_name = name
            bad_name = (not name) or ('SKELETON' in name.upper()) or any(x in name.upper() for x in ['MESH','.PWK','.PRX','SHADER','TEXTURE'])
            if bad_name:
                name=f'{asset_hint}_ANIM_RID_{rid:03d}'
            dur=struct.unpack('>f',D[off+0x1c:off+0x20])[0]
            rot_end=struct.unpack('>I',D[off+0x38:off+0x3c])[0]
            if not(0x58<rot_end<size): continue
            rot_start=struct.unpack('>I',D[off+0x3c:off+0x40])[0]
            if not (0x40 <= rot_start < rot_end): rot_start=0x58
            ts=table_start(off,rot_end,rot_start); scan_end=ts or rot_end
            cand=[]
            rot_start=struct.unpack('>I',D[off+0x3c:off+0x40])[0]
            if not (0x40 <= rot_start < scan_end): rot_start=0x58
            for rel in range(rot_start,max(rot_start,scan_end-4),2):
                tr=decode_explicit(off,rel,scan_end,dur)
                if tr: cand.append(tr)
            explicit=[]; last=0
            for tr in cand:
                if tr['rel']>=last:
                    if explicit and tr['bone']<explicit[-1]['bone']: continue
                    explicit.append(tr); last=tr['end']
            tracks=[]
            for i,tr in enumerate(explicit):
                tracks.append(tr)
                nxt=explicit[i+1]['rel'] if i+1<len(explicit) else None
                if nxt and nxt>tr['end']:
                    # Search short alignments for continuation tracks.
                    best=None
                    for lead in (0,2,4,6):
                        for rem in (0,2,4,6):
                            start2=tr['end']+lead; end2=nxt-rem
                            if end2>start2 and (end2-start2)%8==0:
                                c=decode_cont(off,start2,end2,tr['bone']+1,dur)
                                if c and (best is None or c['count']>best['count']):
                                    best=c
                    if best: tracks.append(best)
            if explicit:
                gap_start=explicit[-1]['end']; gap_end=ts or scan_end
                if gap_end>gap_start:
                    best=None
                    for lead in (0,2,4,6):
                        for rem in (0,2,4,6):
                            start2=gap_start+lead; end2=gap_end-rem
                            if end2>start2 and (end2-start2)%8==0:
                                c=decode_cont(off,start2,end2,explicit[-1]['bone']+1,dur)
                                if c and (best is None or c['count']>best['count']):
                                    best=c
                    if best: tracks.append(best)
            unique=[]; seen=set()
            for tr in sorted(tracks,key=lambda x:(x['rel'],x['bone'])):
                if tr['bone'] not in seen:
                    seen.add(tr['bone']); unique.append(tr)
            if unique:
                safe=re.sub(r'[^A-Za-z0-9_.-]+','_',name)[:120]
                raw_payload=D[off:off+size]
                (rawdir/f'{safe}.bin').write_bytes(raw_payload)
                decoded.append({'rid':rid,'name':name,'duration':dur,'size':size,'rot_section_end':rot_end,'tracks':unique})
                raw_entries.append({
                    'resource_id':rid,
                    'name':name,
                    'descriptor_name':raw_descriptor_name,
                    'safe_filename':f'{safe}.bin',
                    'absolute_offset':hex(off),
                    'size':size,
                    'duration_seconds':dur,
                    'sha256':hashlib.sha256(raw_payload).hexdigest(),
                    'import_rule':'exact_raw_same_size_only'
                })
        except Exception:
            continue

    # Native track dump for animation decode debugging.
    native_dump=[]
    for a in decoded:
        native_dump.append({
            'resource_id':a['rid'], 'name':a['name'], 'duration':a['duration'],
            'rotation_section_end':hex(a['rot_section_end']),
            'tracks':[{'bone_id':tr['bone'], 'bone_name':bone_names[tr['bone']] if tr['bone'] < len(bone_names) else str(tr['bone']),
                       'layout':tr['layout'], 'track_rel':hex(tr['rel']), 'record_count':tr['count'],
                       'records':[k[2] for k in tr['keys']]} for tr in a['tracks']]
        })
    (rawdir/'animation_native_tracks_v11.json').write_text(json.dumps(native_dump, indent=2))
    return decoded, raw_entries

def make_fbx(asset,outdir,vertices,normals,uvs,poly_indices,vertex_weights,skeleton,bone_names,parent,col_global,global_pos,tex_bindings,animations):
    BONE_COUNT=len(bone_names)
    BASE_ID=(int(hashlib.sha1(asset.encode('utf-8')).hexdigest()[:8],16)%1000000000) + 2000000000
    GEOM_ID=BASE_ID+1; MODEL_ID=BASE_ID+2; MAT_ID=BASE_ID+3; SKIN_ID=BASE_ID+4; POSE_ID=BASE_ID+5; TEX_ID_BASE=BASE_ID+100; VID_ID_BASE=BASE_ID+200; BONE_MODEL_BASE=BASE_ID+1000; BONE_ATTR_BASE=BASE_ID+2000; CLUSTER_BASE=BASE_ID+3000; ANIM_ID_BASE=BASE_ID+1000000
    scaled_vertices=[(x*BDG_FBX_EXPORT_SCALE,y*BDG_FBX_EXPORT_SCALE,z*BDG_FBX_EXPORT_SCALE) for x,y,z in vertices]
    def scaled_matrix(m):
        out=[list(row) for row in m]
        out[0][3]*=BDG_FBX_EXPORT_SCALE; out[1][3]*=BDG_FBX_EXPORT_SCALE; out[2][3]*=BDG_FBX_EXPORT_SCALE
        return out
    scaled_col_global={i:scaled_matrix(m) for i,m in col_global.items()}
    scaled_global_pos={i:(p[0]*BDG_FBX_EXPORT_SCALE,p[1]*BDG_FBX_EXPORT_SCALE,p[2]*BDG_FBX_EXPORT_SCALE) for i,p in global_pos.items()}
    geometry=Node('Geometry',[PLong(GEOM_ID),PStr(f'Geometry::{asset}_Geometry'),PStr('Mesh')],[Node('Vertices',[ADouble(flat3(scaled_vertices))]),Node('PolygonVertexIndex',[AInt(poly_indices)]),Node('GeometryVersion',[PInt(124)]),Node('LayerElementNormal',[PInt(0)],[Node('Version',[PInt(101)]),Node('Name',[PStr('')]),Node('MappingInformationType',[PStr('ByPolygonVertex')]),Node('ReferenceInformationType',[PStr('Direct')]),Node('Normals',[ADouble(flat3(normals))])]),Node('LayerElementUV',[PInt(0)],[Node('Version',[PInt(101)]),Node('Name',[PStr('UVChannel_1')]),Node('MappingInformationType',[PStr('ByPolygonVertex')]),Node('ReferenceInformationType',[PStr('Direct')]),Node('UV',[ADouble(flat2(uvs))])]),Node('LayerElementMaterial',[PInt(0)],[Node('Version',[PInt(101)]),Node('Name',[PStr('')]),Node('MappingInformationType',[PStr('AllSame')]),Node('ReferenceInformationType',[PStr('IndexToDirect')]),Node('Materials',[AInt([0])])]),Node('Layer',[PInt(0)],[Node('Version',[PInt(100)]),Node('LayerElement',children=[Node('Type',[PStr('LayerElementNormal')]),Node('TypedIndex',[PInt(0)])]),Node('LayerElement',children=[Node('Type',[PStr('LayerElementUV')]),Node('TypedIndex',[PInt(0)])]),Node('LayerElement',children=[Node('Type',[PStr('LayerElementMaterial')]),Node('TypedIndex',[PInt(0)])])])])
    mesh_model=Node('Model',[PLong(MODEL_ID),PStr(f'Model::{asset}'),PStr('Mesh')],[Node('Version',[PInt(232)]),Node('Properties70',children=[p_node('Lcl Translation','Lcl Translation','','A',0.0,0.0,0.0),p_node('Lcl Rotation','Lcl Rotation','','A',0.0,0.0,0.0),p_node('Lcl Scaling','Lcl Scaling','','A',1.0,1.0,1.0),p_node('DefaultAttributeIndex','int','Integer','',0)]),Node('Shading',[PBool(True)]),Node('Culling',[PStr('CullingOff')])])
    material=Node('Material',[PLong(MAT_ID),PStr(f'Material::{asset}_Material'),PStr('')],[Node('Version',[PInt(102)]),Node('ShadingModel',[PStr('phong')]),Node('MultiLayer',[PInt(0)]),Node('Properties70',children=[p_node('DiffuseColor','Color','','A',0.8,0.8,0.8),p_node('SpecularColor','Color','','A',0.25,0.25,0.25),p_node('BumpFactor','double','Number','A',0.45)])])
    texture_nodes=[]; video_nodes=[]
    for i,(label,rel,prop) in enumerate(tex_bindings):
        tid=TEX_ID_BASE+i; vid=VID_ID_BASE+i; abs_file=str((outdir/rel).resolve())
        # Match the game's repeating texture wrap.
        texture_nodes.append(Node('Texture',[PLong(tid),PStr(f'Texture::{label}'),PStr('')],[Node('Type',[PStr('TextureVideoClip')]),Node('Version',[PInt(202)]),Node('TextureName',[PStr(f'Texture::{label}')]),Node('Properties70',children=[p_node('WrapModeU','enum','','',0),p_node('WrapModeV','enum','','',0),p_node('UseMaterial','bool','','',1),p_node('UseMipMap','bool','','',1)]),Node('Media',[PStr(f'Video::{label}')]),Node('FileName',[PStr(abs_file)]),Node('RelativeFilename',[PStr(rel)]),Node('ModelUVTranslation',[PDouble(0.0),PDouble(0.0)]),Node('ModelUVScaling',[PDouble(1.0),PDouble(1.0)]),Node('Texture_Alpha_Source',[PStr('None')]),Node('Cropping',[PInt(0),PInt(0),PInt(0),PInt(0)])]))
        video_nodes.append(Node('Video',[PLong(vid),PStr(f'Video::{label}'),PStr('Clip')],[Node('Type',[PStr('Clip')]),Node('Properties70',children=[p_node('Path','KString','XRefUrl','',rel)]),Node('UseMipMap',[PInt(0)]),Node('FileName',[PStr(abs_file)]),Node('RelativeFilename',[PStr(rel)])]))
    bone_models=[]; bone_attrs=[]
    for i,name in enumerate(bone_names):
        r=skeleton[i]; tx,ty,tz=r['t']; tx*=BDG_FBX_EXPORT_SCALE; ty*=BDG_FBX_EXPORT_SCALE; tz*=BDG_FBX_EXPORT_SCALE; rx,ry,rz=quat_to_euler_xyz_degrees(r['q']); child_ids=[j for j,p in parent.items() if p==i]
        limb_len=max(0.5,dist(scaled_global_pos[child_ids[0]],scaled_global_pos[i])) if child_ids else 3.0*BDG_FBX_EXPORT_SCALE
        bone_models.append(Node('Model',[PLong(BONE_MODEL_BASE+i),PStr(f'Model::{name}'),PStr('LimbNode')],[Node('Version',[PInt(232)]),Node('Properties70',children=[p_node('Lcl Translation','Lcl Translation','','A',float(tx),float(ty),float(tz)),p_node('Lcl Rotation','Lcl Rotation','','A',float(rx),float(ry),float(rz)),p_node('Lcl Scaling','Lcl Scaling','','A',1.0,1.0,1.0),p_node('RotationOrder','enum','','',0),p_node('LimbLength','double','Number','H',float(limb_len)),p_node('Size','double','Number','',1.0)]),Node('Shading',[PBool(True)]),Node('Culling',[PStr('CullingOff')])]))
        bone_attrs.append(Node('NodeAttribute',[PLong(BONE_ATTR_BASE+i),PStr(f'NodeAttribute::{name}'),PStr('LimbNode')],[Node('TypeFlags',[PStr('Skeleton')]),Node('Properties70',children=[p_node('Size','double','Number','',1.0)])]))
    cluster_indices=collections.defaultdict(list); cluster_weights=collections.defaultdict(list)
    for vi,wts in enumerate(vertex_weights):
        for b,wt in wts:
            if 0<=b<BONE_COUNT and wt>1e-6: cluster_indices[b].append(vi); cluster_weights[b].append(float(wt))
    skin=Node('Deformer',[PLong(SKIN_ID),PStr(f'Deformer::{asset}_Skin'),PStr('Skin')],[Node('Version',[PInt(101)]),Node('Link_DeformAcuracy',[PDouble(50.0)])])
    clusters=[]
    for i in range(BONE_COUNT):
        clusters.append(Node('Deformer',[PLong(CLUSTER_BASE+i),PStr(f'SubDeformer::Cluster_{bone_names[i]}'),PStr('Cluster')],[Node('Version',[PInt(100)]),Node('UserData',[PStr(''),PStr('')]),Node('Indexes',[AInt(cluster_indices.get(i,[]))]),Node('Weights',[ADouble(cluster_weights.get(i,[]))]),Node('Transform',[ADouble(identity())]),Node('TransformLink',[ADouble(fbx_matrix_from_col(scaled_col_global[i]))])]))
    pose_children=[Node('Type',[PStr('BindPose')]),Node('Version',[PInt(100)]),Node('NbPoseNodes',[PInt(BONE_COUNT+1)]),Node('PoseNode',children=[Node('Node',[PLong(MODEL_ID)]),Node('Matrix',[ADouble(identity())])])]
    for i in range(BONE_COUNT): pose_children.append(Node('PoseNode',children=[Node('Node',[PLong(BONE_MODEL_BASE+i)]),Node('Matrix',[ADouble(fbx_matrix_from_col(scaled_col_global[i]))])]))
    pose=Node('Pose',[PLong(POSE_ID),PStr(f'Pose::{asset}_BindPose'),PStr('BindPose')],pose_children)
    animation_objects=[]; animation_connections=[]; anim_curve_count=anim_curve_node_count=anim_stack_count=anim_layer_count=0; anim_manifest=[]
    def p_time(name,value): return Node('P',[PStr(name),PStr('KTime'),PStr('Time'),PStr(''),PLong(int(value))])
    def make_curve(cid,name,times,vals):
        nonlocal anim_curve_count
        anim_curve_count+=1
        if len(times)==1:
            times=[times[0], times[0]+(1.0/60.0)]
            vals=[vals[0], vals[0]]
        n=len(vals)
        return Node('AnimationCurve',[PLong(cid),PStr(f'AnimCurve::{name}'),PStr('')],[Node('Default',[PDouble(0.0)]),Node('KeyVer',[PInt(4008)]),Node('KeyTime',[ALong([int(round(t*FBX_TICKS_PER_SECOND)) for t in times])]),Node('KeyValueFloat',[AFloat([float(v) for v in vals])]),Node('KeyAttrFlags',[AInt([24840])]),Node('KeyAttrDataFloat',[AFloat([0,0,0,0])]),Node('KeyAttrRefCount',[AInt([n])])])

    for ai,a in enumerate(animations):
        sid=ANIM_ID_BASE+ai*10000+1; lid=ANIM_ID_BASE+ai*10000+2
        animation_objects += [Node('AnimationStack',[PLong(sid),PStr(f'AnimStack::{a["name"]}'),PStr('')],[Node('Properties70',children=[p_time('LocalStart',0),p_time('LocalStop',int(round(a['duration']*FBX_TICKS_PER_SECOND))),p_time('ReferenceStart',0),p_time('ReferenceStop',int(round(a['duration']*FBX_TICKS_PER_SECOND)))])]), Node('AnimationLayer',[PLong(lid),PStr(f'AnimLayer::{a["name"]}_Layer'),PStr('')])]
        anim_stack_count+=1; anim_layer_count+=1; animation_connections.append(Node('C',[PStr('OO'),PLong(lid),PLong(sid)])); man_tracks=[]
        for ti,tr in enumerate(a['tracks']):
            bone=tr['bone']; bname=bone_names[bone]; times=[k[0] for k in tr['keys']]; quats=[k[1] for k in tr['keys']]
            if times and times[0]>1e-7: times=[0.0]+times; quats=[quats[0]]+quats
            src_key_count=len(times)
            times,quats=sample_quat_keys(times,quats,a.get('duration',0.0),fps=60.0)
            root_stabilized=False
            fbx_pose_delta_from_bind=False
            if ANIM_PREVIEW_MODE == 'root_stabilized' and bone == 0 and 0 in skeleton:
                quats=stabilize_root_quats_to_bind(quats, skeleton[0]['q'])
                root_stabilized=True
            elif ANIM_PREVIEW_MODE == 'bind_delta':
                bind_q=skeleton[bone]['q'] if bone in skeleton else (0,0,0,1)
                quats=[native_abs_to_blender_pose_delta_quat(q, bind_q) for q in quats]
                fbx_pose_delta_from_bind=True
            eulers=unwrap_eulers([quat_to_euler_xyz_degrees(q) for q in quats])
            cnode_id=ANIM_ID_BASE+ai*10000+100+ti
            cnode=Node('AnimationCurveNode',[PLong(cnode_id),PStr(f'AnimCurveNode::{a["name"]}_{bname}_R'),PStr('')],[Node('Properties70',children=[p_node('d|X','Number','','A',0.0),p_node('d|Y','Number','','A',0.0),p_node('d|Z','Number','','A',0.0)])])
            animation_objects.append(cnode); anim_curve_node_count+=1; animation_connections += [Node('C',[PStr('OO'),PLong(cnode_id),PLong(lid)]),Node('C',[PStr('OP'),PLong(cnode_id),PLong(BONE_MODEL_BASE+bone),PStr('Lcl Rotation')])]
            for axis,idx in [('X',0),('Y',1),('Z',2)]:
                cid=ANIM_ID_BASE+ai*10000+1000+ti*10+idx
                animation_objects.append(make_curve(cid,f'{a["name"]}_{bname}_R_{axis}',times,[e[idx] for e in eulers])); animation_connections.append(Node('C',[PStr('OP'),PLong(cid),PLong(cnode_id),PStr(f'd|{axis}')]))
            man_tracks.append({'bone_id':bone,'bone_name':bname,'layout':tr['layout'],'record_count':tr['count'],'source_key_count':src_key_count,'exported_key_count':len(times),'baked_quaternion_slerp_60fps':True,'fbx_preview_mode':ANIM_PREVIEW_MODE,'fbx_pose_delta_from_bind':fbx_pose_delta_from_bind,'fbx_root_stabilized_to_bind':root_stabilized})
        anim_manifest.append({'resource_id':a['rid'],'name':a['name'],'duration_seconds':a['duration'],'track_count':len(man_tracks),'exported_tracks':man_tracks})
    objects=Node('Objects',children=[geometry,mesh_model,material]+texture_nodes+video_nodes+bone_models+bone_attrs+[skin]+clusters+[pose]+animation_objects)
    con=[Node('C',[PStr('OO'),PLong(MODEL_ID),PLong(0)]),Node('C',[PStr('OO'),PLong(GEOM_ID),PLong(MODEL_ID)]),Node('C',[PStr('OO'),PLong(MAT_ID),PLong(MODEL_ID)])]
    for i,(_,_,prop) in enumerate(tex_bindings): con += [Node('C',[PStr('OP'),PLong(TEX_ID_BASE+i),PLong(MAT_ID),PStr(prop)]),Node('C',[PStr('OO'),PLong(VID_ID_BASE+i),PLong(TEX_ID_BASE+i)])]
    for i in range(BONE_COUNT): con.append(Node('C',[PStr('OO'),PLong(BONE_ATTR_BASE+i),PLong(BONE_MODEL_BASE+i)])); con.append(Node('C',[PStr('OO'),PLong(BONE_MODEL_BASE+i),PLong(BONE_MODEL_BASE+parent[i] if parent[i]>=0 else 0)]))
    con.append(Node('C',[PStr('OO'),PLong(SKIN_ID),PLong(GEOM_ID)]))
    for i in range(BONE_COUNT): con += [Node('C',[PStr('OO'),PLong(CLUSTER_BASE+i),PLong(SKIN_ID)]),Node('C',[PStr('OO'),PLong(BONE_MODEL_BASE+i),PLong(CLUSTER_BASE+i)])]
    con.extend(animation_connections); connections=Node('Connections',children=con)
    def objtype(n,c): return Node('ObjectType',[PStr(n)],[Node('Count',[PInt(c)])])
    definitions=Node('Definitions',children=[Node('Version',[PInt(100)]),Node('Count',[PInt(3+len(texture_nodes)+len(video_nodes)+BONE_COUNT*3+2+len(animation_objects))]),objtype('Model',1+BONE_COUNT),objtype('Geometry',1),objtype('Material',1),objtype('Texture',len(texture_nodes)),objtype('Video',len(video_nodes)),objtype('NodeAttribute',BONE_COUNT),objtype('Deformer',1+BONE_COUNT),objtype('Pose',1),objtype('AnimationStack',anim_stack_count),objtype('AnimationLayer',anim_layer_count),objtype('AnimationCurveNode',anim_curve_node_count),objtype('AnimationCurve',anim_curve_count)])
    global_settings=Node('GlobalSettings',children=[Node('Version',[PInt(1000)]),Node('Properties70',children=[p_node('UpAxis','int','Integer','',2),p_node('UpAxisSign','int','Integer','',1),p_node('FrontAxis','int','Integer','',1),p_node('FrontAxisSign','int','Integer','',-1),p_node('CoordAxis','int','Integer','',0),p_node('CoordAxisSign','int','Integer','',1),p_node('UnitScaleFactor','double','Number','',1.0),p_node('OriginalUnitScaleFactor','double','Number','',1.0)])])
    header=Node('FBXHeaderExtension',children=[Node('FBXHeaderVersion',[PInt(1003)]),Node('FBXVersion',[PInt(7400)]),Node('EncryptionType',[PInt(0)]),Node('Creator',[PStr('BDG to FBX v20 exact-native raw animation import; conservative FBX preview')])])
    takes_children=[Node('Current',[PStr('')])]
    for a in animations:
        ticks=int(round(a['duration']*FBX_TICKS_PER_SECOND)); takes_children.append(Node('Take',[PStr(a['name'])],[Node('FileName',[PStr(f'{a["name"]}.tak')]),Node('LocalTime',[PLong(0),PLong(ticks)]),Node('ReferenceTime',[PLong(0),PLong(ticks)])]))
    f=io.BytesIO(); f.write(b'Kaydara FBX Binary  \x00\x1a\x00'); f.write(struct.pack('<I',7400))
    for n in [header,Node('FileId',[PRaw(b'\0'*16)]),global_settings,definitions,objects,connections,Node('Takes',children=takes_children)]: write_node(f,n)
    f.write(NULL_RECORD); f.write(b'\0'*160)
    (outdir/f'{asset}.fbx').write_bytes(f.getvalue())
    return {'animation_manifest':anim_manifest,'weighted_bones':len([i for i in range(BONE_COUNT) if cluster_indices.get(i)]),'fbx':f'{asset}.fbx'}


def extract_one(base,shape,anim,pvms,root,force=False):
    asset=base.replace(' ','_')
    out=root/f'{base}-Kaiju-Extracted'
    if out.exists():
        if force: shutil.rmtree(out)
        else: raise ValueError(f'Output exists: {out}')
    out.mkdir(parents=True)
    D=shape.read_bytes(); st_off,st_count,strings=find_strtab(D); skel_base,skel_root,skeleton=find_skeleton(D,strings)
    bone_count=max(skeleton)+1; bone_names=[skeleton[i]['name'] for i in range(bone_count)]; parent={i:skeleton[i]['parent'] for i in range(bone_count)}
    col_global={}
    def comp(i):
        if i in col_global: return col_global[i]
        m=col_local(skeleton[i])
        if parent[i]>=0: m=mm(comp(parent[i]),m)
        col_global[i]=m; return m
    for i in range(bone_count): comp(i)
    global_pos={i:(col_global[i][0][3],col_global[i][1][3],col_global[i][2][3]) for i in range(bone_count)}
    submeshes,skipped=choose_meshes(D,bone_count)
    vertices=[]; normals=[]; uvs=[]; vertex_weights=[]; face_count=0; mesh_stats=[]
    for si,sm in enumerate(submeshes):
        for face in sm['faces']:
            ids=[]
            for idx in face:
                off=sm['v_start']+idx*sm['v_stride']
                pos,uv,nrm,wts=parse_vertex_by_layout(D,off,bone_count,sm['layout'])
                p=tuple(map(float,pos)); t=tuple(map(float,uv)); nn=tuple(map(float,nrm))
                ids.append(len(vertices)); vertices.append(p); uvs.append(t); normals.append(nn); vertex_weights.append(wts)
            face_count+=1
        sm_positions=[]
        for vi in range(sm['v_count']):
            off=sm['v_start']+vi*sm['v_stride']
            pos=parse_vertex_by_layout(D,off,bone_count,sm['layout'])[0]
            sm_positions.append(pos)
        bbox={'min':[min(p[i] for p in sm_positions) for i in range(3)],'max':[max(p[i] for p in sm_positions) for i in range(3)]}
        mesh_stats.append({'submesh':si,'layout':sm['layout'],'display_list_start':hex(sm['dl_start']),'display_list_end':hex(sm['dl_end']),'vertex_start':hex(sm['v_start']),'vertex_stride':sm['v_stride'],'vertex_count':sm['v_count'],'triangle_faces':len(sm['faces']),'validation_score':sm['validation_score'],'index_width':sm.get('index_width',6),'bounds':bbox})
    tex_manifest,tex_bindings=decode_textures(D,strings,out,asset)
    animations=[]; animation_resource_locations=[]
    vertices,normals,uvs,vertex_weights,poly_indices,duplicate_seam_faces_skipped = _filter_duplicate_seam_faces(vertices,normals,uvs,vertex_weights)
    face_count = len(poly_indices) // 3
    fbxinfo=make_fbx(asset,out,vertices,normals,uvs,poly_indices,vertex_weights,skeleton,bone_names,parent,col_global,global_pos,tex_bindings,animations)
    manifest={'source_shapes':shape.name,'source_anim':None,'string_table_offset':hex(st_off),'skeleton_base':hex(skel_base),'skeleton_root':hex(skel_root),'bone_count':bone_count,'mesh_stats':mesh_stats,'skipped_mesh_candidates':skipped,'textures':tex_manifest,'animations':fbxinfo['animation_manifest'],'animation_resource_locations':animation_resource_locations,'fbx_export_scale':BDG_FBX_EXPORT_SCALE,'triangles':face_count,'control_points':len(vertices),'weighted_bones':fbxinfo['weighted_bones'],'fbx':fbxinfo['fbx'],'duplicate_seam_faces_skipped':duplicate_seam_faces_skipped,'debug_obj_exports':[],'animation_preview_mode':'disabled_shapes_only','import_scope':'Importer patches same-topology mesh streams and textures by default. Skeleton rest-pose writeback is opt-in with --with-skeleton to avoid Blender FBX axis conversion rotating monsters in-game.'}
    manifest['bones']=[{'idx':i,'name':bone_names[i],'parent':parent[i],'local_translation':skeleton[i]['t'],'local_quaternion_xyzw':skeleton[i]['q'],'global_position':global_pos[i]} for i in range(bone_count)]
    def _sha256_file(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''):
                h.update(chunk)
        return h.hexdigest()
    manifest['file_hashes']={fbxinfo['fbx']:_sha256_file(out/fbxinfo['fbx'])}
    for tex in sorted((out/'textures').glob('*.png')):
        manifest['file_hashes'][f'textures/{tex.name}']=_sha256_file(tex)
    raw_dir=out/'animations_raw'
    if raw_dir.exists():
        for raw in sorted(raw_dir.iterdir()):
            if raw.is_file(): manifest['file_hashes'][f'animations_raw/{raw.name}']=_sha256_file(raw)
    (out/'import_log.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for p in pvms:
        try: shutil.copy2(p,out/p.name)
        except Exception: pass
    return manifest

def main():
    ap=argparse.ArgumentParser(description='BDG to FBX v4 all-kaiju extractor')
    ap.add_argument('folder',nargs='?',default='.')
    ap.add_argument('--all',action='store_true')
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args(); root=Path(clean_arg(args.folder)).resolve(); sets=find_sets(root,args.all)
    reports=[]; errors=[]
    for base,shape,anim,pvms in sets:
        print(f'== Extracting {base} ==')
        try:
            man=extract_one(base,shape,anim,pvms,root,args.force); reports.append({'base':base,'status':'ok','fbx':man['fbx'],'bones':man['bone_count'],'triangles':man['triangles'],'animations':len(man['animations']),'skipped_mesh_candidates':len(man['skipped_mesh_candidates'])})
            print(f'   ok: bones={man["bone_count"]} tris={man["triangles"]} anims={len(man["animations"])} skipped_mesh_candidates={len(man["skipped_mesh_candidates"])}')
        except Exception as e:
            errors.append({'base':base,'status':'error','error':str(e)}); print(f'   ERROR: {e}')
    (root/'all_kaiju_extract_report.json').write_text(json.dumps({'reports':reports,'errors':errors},indent=2),encoding='utf-8')
    if errors: print(f'Finished with {len(errors)} errors. See all_kaiju_extract_report.json')
    else: print('Finished all extracts. See all_kaiju_extract_report.json')
if __name__=='__main__': main()
