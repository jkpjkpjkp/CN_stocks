use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyIterator};

#[pyfunction]
fn fil_and_chunk<'py>(py: Python<'py>, dt: &'py PyAny, x: &'py PyAny) -> PyResult<Py<PyBytes>> {
    // helper to map signed i32 delta into a u8 with offset 64
    let to_u8 = |val: i32| -> u8 {
        if val > -64 && val < 64 {
            (val + 64) as u8
        } else {
            0
        }
    };

    let dt_iter = PyIterator::from_object(dt)?;
    let x_iter = PyIterator::from_object(x)?;

    let mut ret: Vec<u8> = Vec::new();

    // iterate both Python iterators in lockstep
    let mut dt_it = dt_iter;
    let mut x_it = x_iter;
    loop {
        let dt_next = dt_it.next();
        let x_next = x_it.next();

        match (dt_next, x_next) {
            (Some(Err(e)), _) | (_, Some(Err(e))) => return Err(e),
            (Some(Ok(dt_obj)), Some(Ok(x_obj))) => {
                // convert dt_obj to seconds (try int, float, or timedelta.total_seconds())
                let td_secs: u64 = if let Ok(i) = dt_obj.extract::<u64>() {
                    i
                } else if let Ok(f) = dt_obj.extract::<f64>() {
                    f as u64
                } else if let Ok(total_obj) = dt_obj.call_method0("total_seconds") {
                    if let Ok(total) = total_obj.extract::<f64>() {
                        total as u64
                    } else {
                        0
                    }
                } else {
                    // fallback to zero if we can't interpret the item
                    0
                };

                if td_secs > 3600 {
                    ret.push(0);
                } else {
                    // add a 0x40 every 3 seconds, rounded down, minus one as original code intended
                    let repeat = td_secs.saturating_div(3).saturating_sub(1) as usize;
                    if repeat > 0 {
                        ret.extend(std::iter::repeat(0x40u8).take(repeat));
                    }

                    let del: i32 = x_obj.extract()?;
                    ret.push(to_u8(del));
                }
            }
            // if either iterator finished, stop
            _ => break,
        }
    }

    Ok(PyBytes::new(py, &ret).into_py(py))
}

#[pymodule]
fn pyrun(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fil_and_chunk, m)?)?;
    Ok(())
}
