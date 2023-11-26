# yolox_rust

Rust implementation of the [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/tree/main) inference module

## Run
### Single image
```
git clone https://github.com/Rfurukawa3/yolox_rust.git
cd yolox_rust/
cargo build --release
target/release/main_cli assets/images/demo.jpg 
```

The result image is saved to assets/images/demo_visualized.jpg.

### Multi images
```
target/release/main_cli A.jpg B.jpg C.jpg
```

The result images are saved to assets/images/A_visualized.jpg, assets/images/B_visualized.jpg, and assets/images/C_visualized.jpg.  
If `--output` option is not specified, they are saved in assets/images.
