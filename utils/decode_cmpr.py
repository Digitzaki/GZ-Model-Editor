from PIL import Image
import struct, os, sys

def rgb565(c):
    r=((c>>11)&31)*255//31
    g=((c>>5)&63)*255//63
    b=(c&31)*255//31
    return (r,g,b,255)

def decode_cmpr(data,w,h):
    img=Image.new('RGBA',(w,h))
    pos=0
    # CMPR macroblocks 8x8 containing four 4x4 DXT1 blocks in order TL,TR,BL,BR
    for y in range(0,h,8):
      for x in range(0,w,8):
        for by,bx in [(0,0),(0,4),(4,0),(4,4)]:
          if pos+8>len(data): break
          c0,c1,bits=struct.unpack('>HHI',data[pos:pos+8]); pos+=8
          p=[rgb565(c0), rgb565(c1)]
          if c0>c1:
            p.append(tuple((2*p[0][i]+p[1][i])//3 for i in range(3))+(255,))
            p.append(tuple((p[0][i]+2*p[1][i])//3 for i in range(3))+(255,))
          else:
            p.append(tuple((p[0][i]+p[1][i])//2 for i in range(3))+(255,))
            p.append((0,0,0,0))
          for py in range(4):
            for px in range(4):
              idx=(bits >> (30 - 2*(py*4+px))) & 3
              if x+bx+px<w and y+by+py<h:
                img.putpixel((x+bx+px,y+by+py),p[idx])
    return img

if __name__=='__main__':
    fn=sys.argv[1]; off=int(sys.argv[2],0); w=int(sys.argv[3]); h=int(sys.argv[4]); out=sys.argv[5]
    d=open(fn,'rb').read()[off:off+(w*h//2)]
    img=decode_cmpr(d,w,h)
    img.save(out)
