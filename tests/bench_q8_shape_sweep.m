#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float f16f(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff; uint32_t o;
  if(e==0){ if(m==0)o=s<<31; else{ e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; o=(s<<31)|(e<<23)|(m<<13);} }
  else if(e==31) o=(s<<31)|0x7f800000|(m<<13); else o=(s<<31)|((e-15+127)<<23)|(m<<13);
  float r; memcpy(&r,&o,4); return r; }
static uint16_t ff16(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t s=(x>>31)&1,e=(x>>23)&0xff,m=x&0x7fffff;
  if(e==0) return s<<15; int ne=(int)e-127+15; if(ne<=0) return s<<15; if(ne>=31) return (s<<15)|0x7c00;
  return (uint16_t)((s<<15)|(ne<<10)|(m>>13)); }

int main(int argc,char**argv){
@autoreleasepool{
  double peak = argc>1?atof(argv[1]):400.0;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  NSError*err=nil;
  id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"tests/bench_q8_shape_sweep.metallib"] error:&err];
  if(!lib){ printf("lib %s\n",[[err localizedDescription]UTF8String]); return 1;}
  id<MTLCommandQueue> q=[dev newCommandQueue];
  printf("device %s   peak ref %.0f GB/s\n",[[dev name]UTF8String],peak);

  int shapes[][2]={{4096,4096},{2048,8192},{1024,16384},{512,32768},{8192,2048}};
  const char* variants[]={"ds4_1_2","ds4_2_2","ds4_4_2","ds4_8_2","ds4_1_4","ds4_2_4","ds4_4_4",
                          "ds4_1_8","ds4_2_8","ds4_4_8",
                          "lane_1_4","lane_2_4","lane_4_4","lane_1_8","lane_2_8","lane_4_8",
                          "lane_1_16","lane_2_16","lane_4_16","lane_1_32","lane_2_32","lane_4_32"};
  int nvar=sizeof(variants)/sizeof(variants[0]);

  for(int si=0;si<5;si++){
    int K=shapes[si][0],N=shapes[si][1],nb=K/32;
    size_t mat=(size_t)N*nb*34;
    int NW=(int)(3ull*1024*1024*1024/mat); if(NW>96)NW=96; if(NW<8)NW=8;
    size_t arena=(size_t)NW*mat;
    id<MTLBuffer> W=[dev newBufferWithLength:arena options:MTLResourceStorageModeShared];
    uint8_t*wp=(uint8_t*)W.contents;
    unsigned s=12345u;
    for(size_t i=0;i<arena;i+=34){ *(uint16_t*)(wp+i)=ff16(0.01f+(float)((s=s*1103515245u+12345u)>>28)*0.001f);
      for(int j=0;j<32;j++){ s=s*1103515245u+12345u; wp[i+2+j]=(uint8_t)(int8_t)((int)((s>>24)&0x7f)-63); } }
    id<MTLBuffer> X=[dev newBufferWithLength:K*4 options:MTLResourceStorageModeShared];
    float*xp=(float*)X.contents; for(int i=0;i<K;i++) xp[i]=((i%37)-18)*0.05f;
    id<MTLBuffer> O=[dev newBufferWithLength:(size_t)N*4*nvar options:MTLResourceStorageModeShared];
    printf("\n== K=%d N=%d  %.1f MB/mat  NW=%d (%.1f GB arena) ==\n",K,N,mat/1e6,NW,arena/1e9);
    double best_gbs[64]; char ok[64];
    float* ref=malloc((size_t)N*4);
    for(int v=0;v<nvar;v++){
      const char*nm=variants[v];
      int nsg,p2; sscanf(nm+ (nm[0]=='d'?4:5), "%d_%d",&nsg,&p2);
      int isds4 = nm[0]=='d';
      int rows_per_tg = isds4 ? p2 : (32/p2)*nsg;
      if(!isds4 && (nb % p2)) { ok[v]=0; best_gbs[v]=0; continue; }
      if(isds4 && nsg*8 > nb*2) {} // allowed
      id<MTLFunction> f=[lib newFunctionWithName:[NSString stringWithUTF8String:nm]];
      if(!f){ok[v]=0;best_gbs[v]=0;continue;}
      id<MTLComputePipelineState> ps=[dev newComputePipelineStateWithFunction:f error:&err];
      if(!ps){ok[v]=0;best_gbs[v]=0;continue;}
      int ntg=(N+rows_per_tg-1)/rows_per_tg;
      double best=1e30;
      for(int rep=0;rep<6;rep++){
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:ps];
        for(int i=0;i<NW;i++){
          [e setBuffer:W offset:(NSUInteger)i*mat atIndex:0];
          [e setBuffer:X offset:0 atIndex:1];
          [e setBuffer:O offset:(NSUInteger)v*N*4 atIndex:2];
          [e setBytes:&K length:4 atIndex:3];
          [e setBytes:&N length:4 atIndex:4];
          if(isds4)[e setThreadgroupMemoryLength:(NSUInteger)32*p2*4 atIndex:0];
          [e dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(32,nsg,1)];
        }
        [e endEncoding];[cb commit];[cb waitUntilCompleted];
        double t=(cb.GPUEndTime-cb.GPUStartTime)/NW;
        if(t>0&&t<best)best=t;
      }
      best_gbs[v]=mat/1e9/best; ok[v]=1;
      float*op=(float*)((uint8_t*)O.contents+(size_t)v*N*4);
      if(v==0){ memcpy(ref,op,(size_t)N*4); }
      else { double mx=0,rm=0; for(int i=0;i<N;i++){ double d=fabs(op[i]-ref[i]); if(d>mx)mx=d; if(fabs(ref[i])>rm)rm=fabs(ref[i]); }
             if(mx/rm>2e-4){ printf("   !! %s relerr %.2e\n",nm,mx/rm); ok[v]=2; } }
    }
    int idx[64]; for(int i=0;i<nvar;i++)idx[i]=i;
    for(int i=0;i<nvar;i++)for(int j=i+1;j<nvar;j++) if(best_gbs[idx[j]]>best_gbs[idx[i]]){int t=idx[i];idx[i]=idx[j];idx[j]=t;}
    for(int i=0;i<nvar;i++){ int v=idx[i]; if(!ok[v])continue;
      printf("   %6.0f GB/s (%4.1f%%)  %-12s%s\n",best_gbs[v],best_gbs[v]/peak*100,variants[v],ok[v]==2?"  MISMATCH":""); }
    free(ref);
  }
}
return 0;}
