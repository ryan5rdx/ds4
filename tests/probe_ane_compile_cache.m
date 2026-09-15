/*
 * Does the compiled-model cache actually prevent the leak?
 *
 * The rig lost its disk twice to compileModelAtURL:, which builds into a fresh
 * temporary directory on every call and leaves the caller to own it -- 149 GB of
 * shexp_L*.mlmodelc in /var/folders/.../T/ after a day of runs. The fix is in
 * ds4_ane_compile.h: compile once into a stable cache, reuse thereafter.
 *
 * This box has no coremltools, so a real .mlpackage cannot be built here and the
 * FIRST-run compile path cannot be exercised. What can be exercised is the path
 * that actually stops the growth -- every run after the first -- and the
 * staleness rule that decides between them. The test uses a source that is NOT a
 * valid model: on a cache hit that is harmless because Core ML is never invoked,
 * and on a miss compileModelAtURL: fails and returns nil. So the two outcomes are
 * unambiguous, and "did it reuse?" is answered by whether Core ML ran at all.
 */
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <stdio.h>
#include "ds4_ane_compile.h"

static int fails;

static void expect(int cond, const char *what) {
    printf("%s   %s\n", cond ? "ok  " : "FAIL", what);
    if (!cond) fails++;
}

static NSUInteger count_dirs(NSURL *d, NSString *suffix) {
    NSArray *e = [[NSFileManager defaultManager]
                     contentsOfDirectoryAtURL:d
                   includingPropertiesForKeys:nil options:0 error:NULL];
    NSUInteger n = 0;
    for (NSURL *u in e) if ([u.lastPathComponent hasSuffix:suffix]) n++;
    return n;
}

int main(void) { @autoreleasepool {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSURL *root = [[NSURL fileURLWithPath:NSTemporaryDirectory()]
                      URLByAppendingPathComponent:@"ds4-anecache-probe"];
    [fm removeItemAtURL:root error:NULL];
    [fm createDirectoryAtURL:root withIntermediateDirectories:YES
                  attributes:nil error:NULL];

    NSURL *src = [root URLByAppendingPathComponent:@"shexp_L00_fused.mlpackage"];
    [fm createDirectoryAtURL:src withIntermediateDirectories:YES
                  attributes:nil error:NULL];

    setenv("DS4_ANE_COMPILE_CACHE", root.path.UTF8String, 1);
    unsetenv("DS4_ANE_COMPILE_CACHE_REFRESH");

    /* --- miss: no cache entry, so Core ML is invoked and fails on our stub -- */
    NSError *err = nil;
    NSURL *got = ds4_ane_compiled_model_url(src, &err);
    expect(got == nil, "cache MISS reaches Core ML (nil on an invalid package)");

    /* --- hit: plant an entry newer than the source ------------------------- */
    NSURL *cached = [root URLByAppendingPathComponent:@"shexp_L00_fused.mlmodelc"];
    [fm createDirectoryAtURL:cached withIntermediateDirectories:YES
                  attributes:nil error:NULL];
    [fm setAttributes:@{NSFileModificationDate: [NSDate dateWithTimeIntervalSinceNow:+5]}
         ofItemAtPath:cached.path error:NULL];

    err = nil;
    got = ds4_ane_compiled_model_url(src, &err);
    expect(got != nil && [got.path isEqualToString:cached.path],
           "cache HIT returns the cached bundle and never calls Core ML");

    /* --- the leak itself: a hit must not create a temp build --------------- */
    NSURL *tmpdir = [NSURL fileURLWithPath:NSTemporaryDirectory()];
    const NSUInteger before = count_dirs(tmpdir, @".mlmodelc");
    for (int i = 0; i < 20; i++) {
        (void)ds4_ane_compiled_model_url(src, NULL);
    }
    const NSUInteger after = count_dirs(tmpdir, @".mlmodelc");
    expect(after == before,
           "20 further loads add ZERO .mlmodelc bundles to TMPDIR");

    /* --- staleness: a source newer than the cache must recompile ----------- */
    [fm setAttributes:@{NSFileModificationDate: [NSDate dateWithTimeIntervalSinceNow:+60]}
         ofItemAtPath:src.path error:NULL];
    err = nil;
    got = ds4_ane_compiled_model_url(src, &err);
    expect(got == nil, "a source newer than the cache recompiles (not a stale hit)");

    /* --- the refresh override --------------------------------------------- */
    [fm setAttributes:@{NSFileModificationDate: [NSDate dateWithTimeIntervalSinceNow:-60]}
         ofItemAtPath:src.path error:NULL];
    expect(ds4_ane_compiled_model_url(src, NULL) != nil, "stale-free source hits again");
    setenv("DS4_ANE_COMPILE_CACHE_REFRESH", "1", 1);
    expect(ds4_ane_compiled_model_url(src, NULL) == nil,
           "DS4_ANE_COMPILE_CACHE_REFRESH=1 forces a recompile");
    unsetenv("DS4_ANE_COMPILE_CACHE_REFRESH");

    [fm removeItemAtURL:root error:NULL];
    printf("\n%s\n", fails == 0
           ? "PASS: repeated loads reuse one cached bundle and leak nothing"
           : "FAILED");
    return fails == 0 ? 0 : 1;
} }
