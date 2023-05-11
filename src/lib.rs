use std::ffi::c_void;

use dlpark::{
    ffi::{DataType, Device},
    tensor::{
        traits::{AsTensor, HasByteOffset, HasData, HasDevice, HasDtype, HasShape, HasStrides},
        ManagedTensor, ManagerCtx,
    },
};
use image::Rgb;
use pyo3::prelude::*;

struct PyRgbImage(image::RgbImage);

impl HasData for PyRgbImage {
    fn data(&self) -> *mut std::ffi::c_void {
        self.0.as_ptr() as *const c_void as *mut c_void
    }
}

impl HasDevice for PyRgbImage {
    fn device(&self) -> dlpark::ffi::Device {
        Device::CPU
    }
}

impl HasDtype for PyRgbImage {
    fn dtype(&self) -> dlpark::ffi::DataType {
        DataType::U8
    }
}

impl HasShape for PyRgbImage {
    fn shape(&self) -> dlpark::tensor::Shape {
        dlpark::tensor::Shape::Owned(
            [self.0.height(), self.0.width(), 3]
                .map(|x| x as i64)
                .to_vec(),
        )
    }
}

impl HasStrides for PyRgbImage {}
impl HasByteOffset for PyRgbImage {
    fn byte_offset(&self) -> u64 {
        0
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
    PyRgbImage(rgb_img).into()
}

#[pyfunction]
fn write_image(filename: &str, tensor: ManagedTensor) {
    // let v: Vec<u8> = unsafe {
    //     Vec::from_raw_parts(
    //         tensor.data::<u8>() as *mut u8,
    //         tensor.num_elements(),
    //         tensor.num_elements(),
    //     )
    // };

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
