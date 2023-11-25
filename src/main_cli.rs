use image::{self};
use rusttype::{Font, Scale};
use yolox_rust::visualize::draw;
use yolox_rust::yolox::Predictor;

fn main() {
    let class_map: Vec<&str> = vec![
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ];

    let width = 416;
    let height = 416;
    let num_classes = class_map.len() as u32;
    let model_path = "assets/models/yolox_nano.onnx";
    let predictor = Predictor::new(width, height, num_classes, model_path).unwrap();
    let nms_thr = 0.45;
    let score_thr = 0.1;

    let image = image::open("assets/images/demo.jpg").unwrap().to_rgb8();
    let bboxes = predictor.inference(&image, nms_thr, score_thr).unwrap();

    for bbox in &bboxes {
        println!("{:?}", bbox);
    }

    let color = image::Rgb([255, 0, 0]);
    let text_scale = Scale::uniform(32.0);
    let font_data: &[u8] =
        include_bytes!("../assets/fonts/dejavu-sans-ttf-2.37/ttf/DejaVuSans.ttf");
    let font: Font<'static> = Font::try_from_bytes(font_data).unwrap();

    let visualized = draw(&image, &bboxes, &class_map, color, text_scale, &font).unwrap();
    visualized
        .save("assets/images/demo_visualized.jpg")
        .unwrap();
}
