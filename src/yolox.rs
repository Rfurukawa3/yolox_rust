use anyhow::{self, Context};
use image::{imageops, RgbImage};
use std::path::Path;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::ndarray::prelude::*;

const STRIDES: [u32; 3] = [8, 16, 32]; // Strides for each scale.

pub struct Predictor {
    width: u32,
    height: u32,
    num_bboxes: u32,
    num_classes: u32,
    grids: Vec<[u32; 2]>,
    expanded_strides: Vec<u32>,
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    class: u32,
    score: f32,
}

struct PreProcessResult {
    tensor: Tensor,
    ratio: f32,
}

impl Predictor {
    /// Creates a new `Predictor`.
    ///
    /// # Examples
    ///
    /// ```
    /// use yolox_rust::yolox::Predictor;
    ///
    /// let width = 416;
    /// let height = 416;
    /// let num_classes = 80;
    /// let model_path = "models/yolox_nano.onnx";
    /// let predictor = Predictor::new(width, height, num_classes, model_path).unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(
        width: u32,
        height: u32,
        num_classes: u32,
        model_path: P,
    ) -> anyhow::Result<Predictor> {
        // Set up num_bboxes.
        let hsizes = [
            height / STRIDES[0],
            height / STRIDES[1],
            height / STRIDES[2],
        ];
        let wsizes = [width / STRIDES[0], width / STRIDES[1], width / STRIDES[2]];
        let offsets = [
            0,
            hsizes[0] * wsizes[0],
            hsizes[0] * wsizes[0] + hsizes[1] * wsizes[1],
        ];
        let num_bboxes = hsizes[0] * wsizes[0] + hsizes[1] * wsizes[1] + hsizes[2] * wsizes[2];

        // Set up model.
        let model = onnx()
            .model_for_path(model_path)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![1, 3, height as i32, width as i32],
                ),
            )?
            .with_output_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![
                        1,
                        num_bboxes as i32,
                        (num_classes + 5) as i32 // [center_x, center_y, width, height, obj_conf, class0_conf, class1_conf, ...]
                    ],
                ),
            )?
            .into_optimized()?
            .into_runnable()?;

        // Set up grids and expanded_strides for postprocess.
        let mut grids = vec![[0_u32; 2]; num_bboxes as usize];
        let mut expanded_strides = vec![0_u32; num_bboxes as usize];
        for i in 0..STRIDES.len() {
            for y in 0..hsizes[i] {
                for x in 0..wsizes[i] {
                    grids[(offsets[i] + y * wsizes[i] + x) as usize] = [x, y];
                    expanded_strides[(offsets[i] + y * wsizes[i] + x) as usize] = STRIDES[i];
                }
            }
        }

        Ok(Predictor {
            width,
            height,
            num_bboxes,
            num_classes,
            grids,
            expanded_strides,
            model,
        })
    }

    /// Performs inference on the given image.
    ///
    /// # Examples
    ///
    /// ```
    /// use yolox_rust::yolox::Predictor;
    ///
    /// let predictor = Predictor::new(416, 416, 80, "models/yolox_nano.onnx").unwrap();
    /// let image = image::open("images/demo.jpg").unwrap().to_rgb8();
    /// let bboxes = predictor.inference(&image);
    /// ```
    pub fn inference(&self, image: &RgbImage) -> anyhow::Result<Vec<BoundingBox>> {
        let preprocess_result = self.preprocess(image)?;

        let result = self.model.run(tvec!(preprocess_result.tensor.into()))?;
        let output_array = result
            .get(0)
            .context("Inference output is empty.")?
            .to_array_view::<f32>()?;
        let bboxes = self.postprocess(&output_array, preprocess_result.ratio)?;

        Ok(bboxes)
    }

    /// Resize and convert to Tensor.
    ///
    /// The given image is resized to keep the aspect ratio. Blank areas are filled with (114,114,114).
    ///
    /// Reference: https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/data/data_augment.py#L142
    fn preprocess(&self, image: &RgbImage) -> anyhow::Result<PreProcessResult> {
        // r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        // resized_img = cv2.resize(
        //     img,
        //     (int(img.shape[1] * r), int(img.shape[0] * r)),
        //     interpolation=cv2.INTER_LINEAR,
        // ).astype(np.uint8)
        let ratio_w = self.width as f32 / image.width() as f32;
        let ratio_h = self.height as f32 / image.height() as f32;
        let ratio = if ratio_h < ratio_w { ratio_h } else { ratio_w };
        let nwidth = (image.width() as f32 * ratio) as u32;
        let nheight = (image.height() as f32 * ratio) as u32;
        let resized_img = imageops::resize(image, nwidth, nheight, imageops::FilterType::Triangle);

        // padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        let mut padded_img =
            image::RgbImage::from_pixel(self.width, self.height, image::Rgb([114, 114, 114]));
        imageops::replace(&mut padded_img, &resized_img, 0, 0);

        // convert to Tensor
        let padded_tensor = tract_ndarray::Array4::from_shape_fn(
            (1, 3, self.height as usize, self.width as usize),
            |(_, c, y, x)| padded_img[(x as _, y as _)][c] as f32,
        )
        .into();
        Ok(PreProcessResult {
            tensor: padded_tensor,
            ratio: ratio,
        })
    }

    /// Convert model output to BoundingBox.
    ///
    /// Reference: https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/utils/demo_utils.py#L139
    fn postprocess(
        &self,
        output_array: &ArrayViewD<f32>,
        ratio: f32,
    ) -> anyhow::Result<Vec<BoundingBox>> {
        let mut bboxes = Vec::new();

        for bbox in 0..self.num_bboxes {
            let &center_x0 = output_array
                .get([0, bbox as usize, 0])
                .with_context(|| format!("Out of bounds for output_array.get([0,{bbox},0])"))?;
            let &center_y0 = output_array
                .get([0, bbox as usize, 1])
                .with_context(|| format!("Out of bounds for output_array.get([0,{bbox},1])"))?;
            let &box_w0 = output_array
                .get([0, bbox as usize, 2])
                .with_context(|| format!("Out of bounds for output_array.get([0,{bbox},2])"))?;
            let &box_h0 = output_array
                .get([0, bbox as usize, 3])
                .with_context(|| format!("Out of bounds for output_array.get([0,{bbox},3])"))?;

            let center_x = (center_x0 + self.grids[bbox as usize][0] as f32)
                * self.expanded_strides[bbox as usize] as f32;
            let center_y = (center_y0 + self.grids[bbox as usize][1] as f32)
                * self.expanded_strides[bbox as usize] as f32;
            let box_w = box_w0.exp() * self.expanded_strides[bbox as usize] as f32;
            let box_h = box_h0.exp() * self.expanded_strides[bbox as usize] as f32;

            let &obj_conf = output_array
                .get([0, bbox as usize, 4])
                .with_context(|| format!("Out of bounds for output_array.get([0,{bbox},4])"))?;

            let (class_idx, &class_conf) = output_array
                .slice(s![0, bbox as usize, 5..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("output_array contains NaN"))
                .context("num_classes is 0")?;

            let bbox = BoundingBox {
                left: ((center_x - box_w / 2.0) / ratio) as u32,
                top: ((center_y - box_h / 2.0) / ratio) as u32,
                right: ((center_x + box_w / 2.0) / ratio) as u32,
                bottom: ((center_y + box_h / 2.0) / ratio) as u32,
                class: class_idx as u32,
                score: obj_conf * class_conf,
            };
            bboxes.push(bbox);
        }

        Ok(bboxes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let width = 416;
        let height = 416;
        let num_bboxes = 3549;
        let num_classes = 80;
        let model_path = "models/yolox_nano.onnx";
        let predictor = Predictor::new(width, height, num_classes, model_path);
        assert!(predictor.is_ok());

        let predictor = predictor.unwrap();
        assert_eq!(predictor.width, width);
        assert_eq!(predictor.height, height);
        assert_eq!(predictor.num_bboxes, num_bboxes);
        assert_eq!(predictor.num_classes, num_classes);
    }

    #[test]
    fn test_new_no_model() {
        let width = 416;
        let height = 416;
        let num_classes = 80;
        let model_path = "dummy.onnx";
        let predictor = Predictor::new(width, height, num_classes, model_path);
        assert!(predictor.is_err());
    }

    #[test]
    fn test_inference() {
        let width = 416;
        let height = 416;
        let num_classes = 80;
        let model_path = "models/yolox_nano.onnx";
        let predictor = Predictor::new(width, height, num_classes, model_path).unwrap();

        let image = image::open("images/demo.jpg").unwrap().to_rgb8();
        let bboxes = predictor.inference(&image);
        assert!(bboxes.is_ok());
    }
}
