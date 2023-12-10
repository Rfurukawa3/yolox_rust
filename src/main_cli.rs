use anyhow::{self};
use getopts::Options;
use image::{self};
use rusttype::{Font, Scale};
use std::{env, fs, path, process};
use yolox_rust::visualize::{draw, load_class_map};
use yolox_rust::yolox::Predictor;

struct Args {
    inputs: Vec<String>,
    model: String,
    output: String,
    width: u32,
    height: u32,
    nms_thr: f32,
    score_thr: f32,
    class_map: Vec<String>,
}

fn print_usage(program: &str, opts: &Options) {
    let brief = format!("Usage: {} FILE [options]", program);
    print!("{}", opts.usage(&brief));
    process::exit(0);
}

fn parse_args() -> anyhow::Result<Args> {
    const MODEL: &str = "assets/models/yolox_nano.onnx";
    const OUTPUT: &str = "assets/images";
    const WIDTH: u32 = 416;
    const HEIGHT: u32 = 416;
    const NMS_THR: f32 = 0.45;
    const SCORE_THR: f32 = 0.1;
    const CLASS_MAP: &str = "assets/class_map/coco_classes.txt";

    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optopt("m", "model", &format!("default: {MODEL}"), "FILE");
    opts.optopt("o", "output", &format!("default: {OUTPUT}"), "DIR");
    opts.optopt("", "width", &format!("default: {WIDTH}"), "INT");
    opts.optopt("", "height", &format!("default: {HEIGHT}"), "INT");
    opts.optopt("", "nms_thr", &format!("default: {NMS_THR}"), "FLOAT");
    opts.optopt("", "score_thr", &format!("default: {SCORE_THR}"), "FLOAT");
    opts.optopt("", "class_map", &format!("default: {CLASS_MAP}"), "FILE");
    opts.optflag("h", "help", "print this help menu");

    let matches = opts.parse(&args[1..])?;

    if matches.opt_present("h") {
        print_usage(&program, &opts);
    }

    if matches.free.is_empty() {
        print_usage(&program, &opts);
    }

    let model = match matches.opt_str("model") {
        Some(s) => s,
        None => MODEL.to_string(),
    };
    let output = match matches.opt_str("output") {
        Some(s) => s,
        None => OUTPUT.to_string(),
    };
    let width: u32 = match matches.opt_str("width") {
        Some(s) => s.parse()?,
        None => WIDTH,
    };
    let height: u32 = match matches.opt_str("height") {
        Some(s) => s.parse()?,
        None => HEIGHT,
    };
    let nms_thr: f32 = match matches.opt_str("nms_thr") {
        Some(s) => s.parse()?,
        None => NMS_THR,
    };
    let score_thr: f32 = match matches.opt_str("score_thr") {
        Some(s) => s.parse()?,
        None => SCORE_THR,
    };
    let class_map_path = match matches.opt_str("class_map") {
        Some(s) => s,
        None => CLASS_MAP.to_string(),
    };
    let class_map = load_class_map(class_map_path)?;

    Ok(Args {
        inputs: matches.free.clone(),
        model,
        output,
        width,
        height,
        nms_thr,
        score_thr,
        class_map,
    })
}

fn main() {
    let args = parse_args().unwrap();

    let num_classes = args.class_map.len() as u32;
    let predictor = Predictor::new(args.width, args.height, num_classes, args.model).unwrap();

    let color = image::Rgb([255, 0, 0]);
    let text_scale = Scale::uniform(32.0);
    let font_data: &[u8] =
        include_bytes!("../assets/fonts/dejavu-sans-ttf-2.37/ttf/DejaVuSans.ttf");
    let font: Font<'static> = Font::try_from_bytes(font_data).unwrap();

    for input in &args.inputs {
        // if input is a directory, process all images in the directory
        if let Ok(entries) = fs::read_dir(input) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let entry_path = entry.path();
                    if let Ok(image) = image::open(&entry_path) {
                        println!("input image: {}", entry_path.display());
                        let image = image.to_rgb8();
                        let bboxes = predictor
                            .inference(&image, args.nms_thr, args.score_thr)
                            .unwrap();

                        for bbox in &bboxes {
                            println!("{:?}", bbox);
                        }

                        let visualized =
                            draw(&image, &bboxes, &args.class_map, color, text_scale, &font)
                                .unwrap();
                        let basename = entry_path.file_stem().unwrap().to_str().unwrap();
                        let output = format!("{}/{}_visualized.jpg", args.output, basename);
                        visualized.save(output).unwrap();
                    }
                }
            }
            continue;
        }

        // if input is a file, process the file
        let entry_path = path::PathBuf::from(input);
        if let Ok(image) = image::open(&entry_path) {
            println!("input image: {}", entry_path.display());
            let image = image.to_rgb8();
            let bboxes = predictor
                .inference(&image, args.nms_thr, args.score_thr)
                .unwrap();

            for bbox in &bboxes {
                println!("{:?}", bbox);
            }

            let visualized =
                draw(&image, &bboxes, &args.class_map, color, text_scale, &font).unwrap();
            let basename = entry_path.file_stem().unwrap().to_str().unwrap();
            let output = format!("{}/{}_visualized.jpg", args.output, basename);
            visualized.save(output).unwrap();
            continue;
        }

        eprintln!("failed : {}", input)
    }
}
