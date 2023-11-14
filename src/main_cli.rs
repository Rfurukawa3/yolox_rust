use image::{self};
use yolox_rust::yolox::Predictor;

fn main() {
    let width = 416;
    let height = 416;
    let num_classes = 80;
    let model_path = "models/yolox_nano.onnx";
    let predictor = Predictor::new(width, height, num_classes, model_path).unwrap();

    let image = image::open("images/demo.jpg").unwrap().to_rgb8();
    let bboxes = predictor.inference(&image).unwrap();

    println!("{:?}", bboxes[0]);
    println!("{:?}", bboxes[1]);
}
