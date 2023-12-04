use criterion::*;
use image::RgbImage;
use yolox_rust::bbox::BoundingBox;
use yolox_rust::yolox::Predictor;

const MODEL: &str = "assets/models/yolox_nano.onnx";
const WIDTH: u32 = 416;
const HEIGHT: u32 = 416;
const NUM_CLASSES: u32 = 80;
const NMS_THR: f32 = 0.45;
const SCORE_THR: f32 = 0.1;
const IMAGE: &str = "assets/images/demo.jpg";

fn predictor_new(width: u32, height: u32, num_classes: u32, model_path: &str) -> Predictor {
    Predictor::new(width, height, num_classes, model_path).unwrap()
}

fn predictor_inference(predictor: &Predictor, image: &RgbImage) -> Vec<BoundingBox> {
    predictor.inference(image, NMS_THR, SCORE_THR).unwrap()
}

fn bench_new(c: &mut Criterion) {
    c.bench_function("predictor_new", |b| {
        b.iter(|| {
            predictor_new(
                black_box(WIDTH),
                black_box(HEIGHT),
                black_box(NUM_CLASSES),
                black_box(MODEL),
            )
        })
    });
}

fn bench_inference(c: &mut Criterion) {
    let predictor = predictor_new(WIDTH, HEIGHT, NUM_CLASSES, MODEL);
    let image = image::open(IMAGE).unwrap().to_rgb8();
    c.bench_function("predictor_inference", |b| {
        b.iter(|| predictor_inference(&predictor, black_box(&image)))
    });
}

criterion_group!(benches, bench_new, bench_inference);
criterion_main!(benches);
