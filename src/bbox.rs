use anyhow::{self};

#[derive(Debug, Clone)]
pub struct BoundingBox {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    class: u32,
    score: f32,
}

impl BoundingBox {
    pub fn new(
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        class: u32,
        score: f32,
    ) -> anyhow::Result<BoundingBox> {
        let bbox = BoundingBox {
            left,
            top,
            right,
            bottom,
            class,
            score,
        };
        bbox.check()?;
        Ok(bbox)
    }

    /// Checks if the bounding box is valid.
    fn check(&self) -> anyhow::Result<()> {
        if self.left >= self.right {
            anyhow::bail!("left must be less than right.");
        }
        if self.top >= self.bottom {
            anyhow::bail!("top must be less than bottom.");
        }
        Ok(())
    }

    /// Returns the intersection over union (IoU) of two bounding boxes.
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let area1 = (self.right as f32 - self.left as f32) * (self.bottom as f32 - self.top as f32);
        let area2 =
            (other.right as f32 - other.left as f32) * (other.bottom as f32 - other.top as f32);
        let intersection_area =
            (self.right.min(other.right) as f32 - self.left.max(other.left) as f32).max(0.0)
                * (self.bottom.min(other.bottom) as f32 - self.top.max(other.top) as f32).max(0.0);

        intersection_area / (area1 + area2 - intersection_area)
    }

    /// Returns the left coordinate of the bounding box.
    pub fn left(&self) -> u32 {
        self.left
    }

    /// Returns the top coordinate of the bounding box.
    pub fn top(&self) -> u32 {
        self.top
    }

    /// Returns the right coordinate of the bounding box.
    pub fn right(&self) -> u32 {
        self.right
    }

    /// Returns the bottom coordinate of the bounding box.
    pub fn bottom(&self) -> u32 {
        self.bottom
    }

    /// Returns the class index of the bounding box.
    pub fn class(&self) -> u32 {
        self.class
    }

    /// Returns the score of the bounding box.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Returns the width of the bounding box.
    pub fn width(&self) -> u32 {
        self.right - self.left
    }

    /// Returns the height of the bounding box.
    pub fn height(&self) -> u32 {
        self.bottom - self.top
    }
}
