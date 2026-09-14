#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
int main(int argc, const char **argv) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *e = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&e];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&e];
    if (!lib) { printf("COMPILE FAILED: %s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLFunction> f = [lib newFunctionWithName:@"ds4_probe_dq_q4k_equiv"];
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
    if (!p) { printf("PIPELINE FAILED: %s\n", e.localizedDescription.UTF8String); return 1; }
    const NSUInteger BLK = 144, N = 4096;
    id<MTLBuffer> blocks = [dev newBufferWithLength:BLK*N options:MTLResourceStorageModeShared];
    unsigned char *bp = blocks.contents;
    srandom(12345);
    for (NSUInteger i = 0; i < BLK*N; i++) bp[i] = (unsigned char)(random() & 0xff);
    id<MTLBuffer> mm = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> fb = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    memset(mm.contents, 0, 4); memset(fb.contents, 0, 16);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:blocks offset:0 atIndex:0];
    [enc setBuffer:mm offset:0 atIndex:1];
    [enc setBuffer:fb offset:0 atIndex:2];
    [enc setThreadgroupMemoryLength:BLK atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) { printf("cb failed: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
    const unsigned n = *(unsigned*)mm.contents; const float *bad = fb.contents;
    printf("blocks=%lu values=%lu mismatches=%u\n", (unsigned long)N, (unsigned long)N*16*16, n);
    if (n) printf("  first: device=%.9g threadgroup=%.9g il=%d i=%d\n", bad[0], bad[1], (int)bad[2], (int)bad[3]);
    printf("%s\n", n==0 ? "PASS: device and threadgroup Q4_K dequantisers are bit-identical" : "FAILED");
    return n==0?0:1; } }
