use std::ffi::c_void;

use dlpark::prelude::*;
use image::Rgb;
use pyo3::prelude::*;

struct PyRgbImage(image::RgbImage);

impl ToTensor for PyRgbImage {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.0.as_ptr() as *const c_void as *mut c_void
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn shape(&self) -> CowIntArray {
        CowIntArray::from_owned(
            [self.0.height(), self.0.width(), 3]
                .map(|x| x as i64)
                .to_vec(),
        )
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn dtype(&self) -> DataType {
        DataType::U8
    }

    fn strides(&self) -> Option<CowIntArray> {
        None
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn read_image(filename: &str) -> ManagerCtx<PyRgbImage> {
    let img = image::open(filename).unwrap();
    let rgb_img = img.to_rgb8();
    ManagerCtx::new(PyRgbImage(rgb_img))
}

#[pyfunction]
fn write_image(filename: &str, tensor: ManagedTensor) {
    let buf = tensor.as_slice::<u8>();

    let rgb_img = image::ImageBuffer::<Rgb<u8>, _>::from_raw(
        tensor.shape()[1] as u32,
        tensor.shape()[0] as u32,
        buf,
    )
    .unwrap();

    rgb_img.save(filename).unwrap();
}

/// A Python module implemented in Rust.
#[pymodule]
fn dlparkimg(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    m.add_function(wrap_pyfunction!(write_image, m)?)?;
    Ok(())
}
