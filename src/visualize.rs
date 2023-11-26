use crate::bbox::BoundingBox;
use anyhow::{self, Context};
use image::RgbImage;
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect as ProcRect;
use rusttype::{Font, Scale};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Draws bounding boxes on an image.
pub fn draw<T: AsRef<str>>(
    image: &RgbImage,
    bboxes: &Vec<BoundingBox>,
    class_map: &Vec<T>,
    color: image::Rgb<u8>,
    text_scale: Scale,
    font: &Font<'static>,
) -> anyhow::Result<RgbImage> {
    let mut drawn = image.clone();
    for bbox in bboxes {
        draw_hollow_rect_mut(
            &mut drawn,
            ProcRect::at(bbox.left() as i32, bbox.top() as i32)
                .of_size(bbox.width(), bbox.height()),
            color,
        );
        let class = bbox.class() as usize;
        let label = class_map
            .get(class)
            .with_context(|| format!("Out of bounds for class_map.get({class})"))?
            .as_ref();

        let text = format!("{}: {:.2}", label, bbox.score());
        draw_text_mut(
            &mut drawn,
            color,
            bbox.left() as i32,
            bbox.bottom() as i32,
            text_scale,
            font,
            text.as_str(),
        );
    }

    Ok(drawn)
}

/// Loads a class map from a file.
/// The file should contain one class per line.
pub fn load_class_map<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<String>> {
    let file = File::open(path)?;
    let buffered = BufReader::new(file);
    let mut class_map = Vec::new();

    for read in buffered.lines() {
        let line = read?;
        if line == "" {
            continue;
        }
        class_map.push(line);
    }
    Ok(class_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::BoundingBox;
    use image::RgbImage;
    use rusttype::{Font, Scale};

    #[test]
    fn test_draw() {
        let image = RgbImage::new(800, 600);
        let bboxes = vec![
            BoundingBox::new(100, 50, 200, 120, 0, 0.9).unwrap(),
            BoundingBox::new(300, 250, 400, 400, 1, 0.8).unwrap(),
        ];
        let class_map = vec!["cat", "dog"];
        let color = image::Rgb([255, 0, 0]);
        let text_scale = Scale::uniform(32.0);
        let font_data: &[u8] =
            include_bytes!("../assets/fonts/dejavu-sans-ttf-2.37/ttf/DejaVuSans.ttf");
        let font: Font<'static> = Font::try_from_bytes(font_data).unwrap();

        let visualized = draw(&image, &bboxes, &class_map, color, text_scale, &font);
        assert!(visualized.is_ok());

        let visualized = visualized.unwrap();
        assert_eq!(visualized.width(), 800);
        assert_eq!(visualized.height(), 600);
    }
}
