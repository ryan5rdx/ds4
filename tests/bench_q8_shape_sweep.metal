#include <metal_stdlib>
using namespace metal;

// ---- ds4 / llama.cpp shape: NR0 rows per threadgroup, NSG simdgroups split K ----
template<short NSG, short NR0>
void ds4_impl(device const uchar* w, device const float* x, device float* out,
              constant int& K, constant int& N,
              uint tgid, ushort tiisg, ushort sgitg, threadgroup float* sh) {
    const short NQ = 8;
    const int nb = K/32;
    const int r0 = tgid*NR0;
    threadgroup float* shr[NR0];
    float sumf[NR0];
    for (short r=0;r<NR0;++r){ shr[r]=sh+32*r; sumf[r]=0.0f; if(sgitg==0) shr[r][tiisg]=0.0f; }
    const short ix = tiisg/(32/NQ);
    const short il = tiisg%(32/NQ);
    for (int ib = sgitg*NQ+ix; ib < nb; ib += NSG*NQ) {
        float yl[NQ];
        for (short i=0;i<NQ;++i) yl[i]=x[ib*32+il*NQ+i];
        for (short r=0;r<NR0;++r) {
            if (r0+r>=N) break;
            device const uchar* b = w + ((size_t)(r0+r)*(size_t)nb + ib)*34;
            half d = *((device const half*)b);
            device const char* q = (device const char*)(b+2)+il*NQ;
            float s=0.0f;
            for (short i=0;i<NQ;++i) s += (float)q[i]*yl[i];
            sumf[r] += s*(float)d;
        }
    }
    for (short r=0;r<NR0;++r) sumf[r]=simd_sum(sumf[r]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short r=0;r<NR0;++r) if (tiisg==0) shr[r][sgitg]=sumf[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short r=0;r<NR0;++r) {
        float tot = simd_sum(shr[r][tiisg]);
        if (tiisg==0 && sgitg==0 && r0+r<N) out[r0+r]=tot;
    }
}

// ---- MLX-style: KL lanes reduce K for one row; 32/KL rows per simdgroup ----
template<short NSG, short KL>
void lane_impl(device const uchar* w, device const float* x, device float* out,
               constant int& K, constant int& N,
               uint tgid, ushort tiisg, ushort sgitg) {
    const short RPS = 32/KL;
    const int nb = K/32;
    const int rows_per_tg = RPS*NSG;
    const int row = tgid*rows_per_tg + sgitg*RPS + (tiisg/KL);
    const short l = tiisg%KL;
    const int bpl = nb/KL;
    float acc = 0.0f;
    if (row < N) {
        device const uchar* wr = w + (size_t)row*(size_t)nb*34;
        for (int ib = l*bpl; ib < (l+1)*bpl; ++ib) {
            device const uchar* b = wr + (size_t)ib*34;
            half d = *((device const half*)b);
            device const char* q = (device const char*)(b+2);
            device const float* y = x + ib*32;
            float s=0.0f;
            for (short i=0;i<32;++i) s += (float)q[i]*y[i];
            acc += s*(float)d;
        }
    }
    for (short off=KL/2; off>=1; off>>=1) acc += simd_shuffle_down(acc, (ushort)off);
    if (l==0 && row<N) out[row]=acc;
}

#define DS4K(nsg,nr0) \
kernel void ds4_##nsg##_##nr0(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], \
  device float* out [[buffer(2)]], constant int& K [[buffer(3)]], constant int& N [[buffer(4)]], \
  threadgroup float* sh [[threadgroup(0)]], uint tgid [[threadgroup_position_in_grid]], \
  ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
  ds4_impl<nsg,nr0>(w,x,out,K,N,tgid,tiisg,sgitg,sh); }

#define LANEK(nsg,kl) \
kernel void lane_##nsg##_##kl(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], \
  device float* out [[buffer(2)]], constant int& K [[buffer(3)]], constant int& N [[buffer(4)]], \
  uint tgid [[threadgroup_position_in_grid]], \
  ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
  lane_impl<nsg,kl>(w,x,out,K,N,tgid,tiisg,sgitg); }

DS4K(1,2) DS4K(2,2) DS4K(4,2) DS4K(8,2)
DS4K(1,4) DS4K(2,4) DS4K(4,4)
DS4K(1,8) DS4K(2,8) DS4K(4,8)
LANEK(1,4) LANEK(2,4) LANEK(4,4)
LANEK(1,8) LANEK(2,8) LANEK(4,8)
LANEK(1,16) LANEK(2,16) LANEK(4,16)
LANEK(1,32) LANEK(2,32) LANEK(4,32)
