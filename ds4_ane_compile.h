/*
 * Compiled-model cache for the ANE sidecar.
 *
 * THE LEAK. -[MLModel compileModelAtURL:] compiles into a FRESH temporary
 * directory on every call and makes the caller responsible for the result --
 * Apple's contract is that you move it somewhere permanent or delete it.
 * Neither ds4_ane.m nor ds4_ane_helper.m did either, so every ANE-sidecar run
 * left a whole 42-layer set of shexp_L*.mlmodelc bundles behind in
 * /var/folders/.../T/ under a new UUID. Measured on the rig at ~149 GB after a
 * day of repeated runs; it filled lanfear's disk twice and voided the run in
 * progress both times, including PERFONLY3's first attempt.
 *
 * THE FIX, and why it is a cache rather than a delete. Deleting at exit would
 * stop the growth but keep paying ~24 s of compilation per run, and a crashed
 * or SIGKILLed process (which is how the harness stops the worker) would leak
 * anyway. Compiling ONCE into a stable path beside the .mlpackage fixes both:
 * later runs skip compilation entirely, and there is exactly one set on disk no
 * matter how many runs execute.
 *
 * The temp artifact is MOVED into place, so nothing is left behind even on the
 * first run. If the move fails -- a read-only model directory, a different
 * filesystem -- the temp URL is used and registered for deletion at teardown,
 * which is strictly better than abandoning it.
 *
 * Staleness is by modification date against the source package, with
 * DS4_ANE_COMPILE_CACHE_REFRESH=1 as the override. A .mlpackage is a directory
 * and a tool that rewrites only an inner file without touching the directory
 * mtime would not be noticed -- the generator rewrites the package wholesale,
 * so this holds today, and the override exists for when it does not.
 */
#pragma once
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

static inline NSMutableArray<NSURL *> *ds4_ane_compile_temp_registry(void) {
    static NSMutableArray<NSURL *> *reg;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ reg = [NSMutableArray array]; });
    return reg;
}

static inline NSURL *ds4_ane_compile_cache_dir(NSURL *src) {
    const char *e = getenv("DS4_ANE_COMPILE_CACHE");
    NSURL *dir = (e && e[0])
        ? [NSURL fileURLWithPath:@(e)]
        : [[src URLByDeletingLastPathComponent]
              URLByAppendingPathComponent:@".mlmodelc-cache"];
    [[NSFileManager defaultManager] createDirectoryAtURL:dir
                             withIntermediateDirectories:YES
                                              attributes:nil
                                                   error:NULL];
    return dir;
}

static inline NSDate *ds4_ane_mtime(NSURL *u) {
    /* NSURL CACHES resource values. Without this the staleness check reads
     * whatever the mtime was the first time anyone asked, so a rebuilt
     * .mlpackage silently keeps serving the old compiled bundle -- caught by
     * tests/probe_ane_compile_cache.m, which reuses one NSURL across calls the
     * way a long-lived loader would. */
    [u removeCachedResourceValueForKey:NSURLContentModificationDateKey];
    NSDate *d = nil;
    if (![u getResourceValue:&d forKey:NSURLContentModificationDateKey error:NULL]) {
        return nil;
    }
    return d;
}

/* Returns a compiled .mlmodelc URL for `src`, reusing the cached one when it is
 * present and not older than the source. Never returns a URL that nothing owns:
 * either it lives in the cache, or it is registered for cleanup. */
static inline NSURL *ds4_ane_compiled_model_url(NSURL *src, NSError **err) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSURL *cache_dir = ds4_ane_compile_cache_dir(src);
    NSString *stem = [[src lastPathComponent] stringByDeletingPathExtension];
    NSURL *cached = [cache_dir URLByAppendingPathComponent:
                        [stem stringByAppendingPathExtension:@"mlmodelc"]];

    const char *refresh = getenv("DS4_ANE_COMPILE_CACHE_REFRESH");
    const BOOL force = (refresh && refresh[0] && refresh[0] != '0');

    if (!force && [fm fileExistsAtPath:cached.path]) {
        NSDate *cs = ds4_ane_mtime(src), *cc = ds4_ane_mtime(cached);
        if (!cs || !cc || [cc compare:cs] != NSOrderedAscending) return cached;
    }

    NSURL *tmp = [MLModel compileModelAtURL:src error:err];
    if (!tmp) return nil;

    /* Replace atomically where possible; a stale cache entry must not survive a
     * recompile, and two ranks on one host can race here. */
    NSURL *installed = nil;
    [fm removeItemAtURL:cached error:NULL];
    if ([fm moveItemAtURL:tmp toURL:cached error:NULL]) {
        installed = cached;
    } else {
        /* Could not install it. Use the temp build, but own it. */
        [ds4_ane_compile_temp_registry() addObject:tmp];
        installed = tmp;
        static int warned;
        if (!warned) {
            warned = 1;
            fprintf(stderr,
                    "ds4: ANE compiled-model cache unavailable at %s -- using "
                    "per-run temporary builds, removed at teardown\n",
                    cache_dir.path.UTF8String);
        }
    }
    return installed;
}

/* Remove only what this process could not install. The cache itself is meant to
 * persist -- that is the point -- so it is never touched here. */
static inline void ds4_ane_compile_cache_cleanup(void) {
    NSMutableArray<NSURL *> *reg = ds4_ane_compile_temp_registry();
    NSFileManager *fm = [NSFileManager defaultManager];
    for (NSURL *u in reg) [fm removeItemAtURL:u error:NULL];
    [reg removeAllObjects];
}

#endif /* __OBJC__ */
