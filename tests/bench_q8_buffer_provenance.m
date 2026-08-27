#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc,char**argv){
@autoreleasepool{
  double peak = argc>1?atof(argv[1]):400.0;
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError*err=nil;
  id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"tests/bench_q8_shape_sweep.metallib"] error:&err];
  id<MTLCommandQueue> q=[dev newCommandQueue];
  int K=4096,N=4096,nb=K/32; size_t mat=(size_t)N*nb*34; int NW=96; size_t arena=(size_t)NW*mat;
  printf("device %s  K=%d N=%d  mat %.1f MB  arena %.2f GB\n",[[dev name]UTF8String],K,N,mat/1e6,arena/1e9);

  // A: device-allocated shared buffer
  id<MTLBuffer> Wdev=[dev newBufferWithLength:arena options:MTLResourceStorageModeShared];
  memset(Wdev.contents,0x11,arena);

  // B: anonymous mmap wrapped no-copy (what ds4 does for the model map)
  void* m=mmap(NULL,arena,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
  memset(m,0x11,arena);
  id<MTLBuffer> Wmap=[dev newBufferWithBytesNoCopy:m length:arena options:MTLResourceStorageModeShared deallocator:nil];

  // C: file-backed mmap wrapped no-copy (what ds4 actually does with the GGUF)
  const char* path="/tmp/ds4_provenance_arena.bin";
  int fd=open(path,O_RDWR|O_CREAT|O_TRUNC,0644);
  ftruncate(fd,(off_t)arena);
  void* fm=mmap(NULL,arena,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  memset(fm,0x11,arena);
  msync(fm,arena,MS_SYNC);
  id<MTLBuffer> Wfile=[dev newBufferWithBytesNoCopy:fm length:arena options:MTLResourceStorageModeShared deallocator:nil];

  id<MTLBuffer> X=[dev newBufferWithLength:K*4 options:MTLResourceStorageModeShared];
  float*xp=(float*)X.contents; for(int i=0;i<K;i++) xp[i]=((i%37)-18)*0.05f;
  id<MTLBuffer> O=[dev newBufferWithLength:(size_t)N*4 options:MTLResourceStorageModeShared];

  id<MTLFunction> f=[lib newFunctionWithName:@"ds4_4_2"];
  id<MTLComputePipelineState> ps=[dev newComputePipelineStateWithFunction:f error:&err];
  struct { const char*nm; id<MTLBuffer> b; } arms[3]={{"device MTLBuffer",Wdev},{"anon mmap no-copy",Wmap},{"file mmap no-copy",Wfile}};

  // interleaved best-of
  double best[3]={1e30,1e30,1e30};
  for(int rep=0;rep<8;rep++){
    for(int a=0;a<3;a++){
      id<MTLCommandBuffer> cb=[q commandBuffer];
      id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
      [e setComputePipelineState:ps];
      for(int i=0;i<NW;i++){
        [e setBuffer:arms[a].b offset:(NSUInteger)i*mat atIndex:0];
        [e setBuffer:X offset:0 atIndex:1]; [e setBuffer:O offset:0 atIndex:2];
        [e setBytes:&K length:4 atIndex:3]; [e setBytes:&N length:4 atIndex:4];
        [e setThreadgroupMemoryLength:32*2*4 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(N/2,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
      }
      [e endEncoding];[cb commit];[cb waitUntilCompleted];
      double t=(cb.GPUEndTime-cb.GPUStartTime)/NW;
      if(t>0&&t<best[a])best[a]=t;
    }
  }
  for(int a=0;a<3;a++) printf("  %-20s %7.1f us  %6.0f GB/s (%4.1f%%)\n",arms[a].nm,best[a]*1e6,mat/1e9/best[a],mat/1e9/best[a]/peak*100);
  unlink(path);
}
return 0;}
