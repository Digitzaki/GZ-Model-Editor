from PIL import Image

def rgb565(c):
    r=((c>>11)&31)*255//31; g=((c>>5)&63)*255//63; b=(c&31)*255//31
    return r,g,b,255

def rgb5a3(c):
    if c & 0x8000:
        r=((c>>10)&31)*255//31; g=((c>>5)&31)*255//31; b=(c&31)*255//31; a=255
    else:
        a=((c>>12)&7)*255//7; r=((c>>8)&15)*255//15; g=((c>>4)&15)*255//15; b=(c&15)*255//15
    return r,g,b,a

def decode_i4(data,w,h):
    img=Image.new('RGBA',(w,h)); pix=img.load(); pos=0
    for ty in range(0,h,8):
      for tx in range(0,w,8):
        for y in range(8):
          for xpair in range(4):
            if pos>=len(data): break
            byte=data[pos]; pos+=1
            for j,nib in enumerate([(byte>>4)&15,byte&15]):
              x=tx+xpair*2+j; yy=ty+y
              if x<w and yy<h:
                v=nib*17; pix[x,yy]=(v,v,v,255)
    return img

def decode_i8(data,w,h):
    img=Image.new('RGBA',(w,h)); pix=img.load(); pos=0
    for ty in range(0,h,4):
      for tx in range(0,w,8):
        for y in range(4):
          for x in range(8):
            if pos>=len(data): break
            v=data[pos]; pos+=1
            xx=tx+x; yy=ty+y
            if xx<w and yy<h: pix[xx,yy]=(v,v,v,255)
    return img

def decode_ia4(data,w,h):
    img=Image.new('RGBA',(w,h)); pix=img.load(); pos=0
    for ty in range(0,h,4):
      for tx in range(0,w,8):
        for y in range(4):
          for x in range(8):
            if pos>=len(data): break
            b=data[pos]; pos+=1
            a=((b>>4)&15)*17; v=(b&15)*17
            xx=tx+x; yy=ty+y
            if xx<w and yy<h: pix[xx,yy]=(v,v,v,a)
    return img

def decode_ia8(data,w,h):
    img=Image.new('RGBA',(w,h)); pix=img.load(); pos=0
    for ty in range(0,h,4):
      for tx in range(0,w,4):
        for y in range(4):
          for x in range(4):
            if pos+1>=len(data): break
            a=data[pos]; v=data[pos+1]; pos+=2
            xx=tx+x; yy=ty+y
            if xx<w and yy<h: pix[xx,yy]=(v,v,v,a)
    return img

def decode_rgb565(data,w,h):
    img=Image.new('RGBA',(w,h)); pix=img.load(); pos=0
    for ty in range(0,h,4):
      for tx in range(0,w,4):
        for y in range(4):
          for x in range(4):
            if pos+1>=len(data): break
            c=(data[pos]<<8)|data[pos+1]; pos+=2
            xx=tx+x; yy=ty+y
            if xx<w and yy<h: pix[xx,yy]=rgb565(c)
    return img

def decode_rgb5a3(data,w,h):
    img=Image.new('RGBA',(w,h)); pix=img.load(); pos=0
    for ty in range(0,h,4):
      for tx in range(0,w,4):
        for y in range(4):
          for x in range(4):
            if pos+1>=len(data): break
            c=(data[pos]<<8)|data[pos+1]; pos+=2
            xx=tx+x; yy=ty+y
            if xx<w and yy<h: pix[xx,yy]=rgb5a3(c)
    return img

def decode_rgba8(data,w,h):
    img=Image.new('RGBA',(w,h)); pix=img.load(); pos=0
    for ty in range(0,h,4):
      for tx in range(0,w,4):
        # 16 AR pairs then 16 GB pairs
        ar=[]; gb=[]
        for i in range(16):
            if pos+1 < len(data): ar.append((data[pos],data[pos+1])); pos+=2
        for i in range(16):
            if pos+1 < len(data): gb.append((data[pos],data[pos+1])); pos+=2
        for i in range(16):
            x=i%4; y=i//4; xx=tx+x; yy=ty+y
            if i<len(ar) and i<len(gb) and xx<w and yy<h:
              a,r=ar[i]; g,b=gb[i]; pix[xx,yy]=(r,g,b,a)
    return img
