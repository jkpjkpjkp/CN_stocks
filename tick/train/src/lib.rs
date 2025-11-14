use pyo3::prelude::*;
use pyo3::{Python, PyResult, IntoPyObject, types::PyAnyMethods};
use pyo3::ffi::c_str;
use pyo3::pyfunction
use std::time::Duration;

#[pymodule]
mod pyrun {

    #[pyfunction]
    fn fil_and_chunk(dt: &PyIterator<Duration>, x: &PyIterator<i32>) -> PyResult<()> {
        Python::attach(|py| {
            let to_u8 = |x: i32| -> u8 {
                if x > -64 && x < 64 {
                    (x + 64) as u8
                } else {
                    0
                }
            };
            
            let mut ret: Vec<u8> = Vec::new();
            for (td, del) in dt.zip(x) {
                if td > Duration::from_hours(1) {
                    ret.push(0)
                } else {
                    // we add a '0x40' every 3 seconds, rounded down
                    ret.extend([0x40].repeat(td.as_secs() as usize / 3 - 1));
                    ret.push(to_u8(del))
                }
            }

            Ok(ret)
        })
}
}
