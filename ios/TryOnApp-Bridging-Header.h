#ifndef TryOnApp_Bridging_Header_h
#define TryOnApp_Bridging_Header_h

#if __has_include("../include/rust_tryon_engine.h")
#import "../include/rust_tryon_engine.h"
#elif __has_include("../../include/rust_tryon_engine.h")
#import "../../include/rust_tryon_engine.h"
#else
#import "rust_tryon_engine.h"
#endif

#endif /* TryOnApp_Bridging_Header_h */
