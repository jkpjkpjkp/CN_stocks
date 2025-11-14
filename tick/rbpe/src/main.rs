use std::fs;
use std::io;

mod bpe;

fn read_binary_file_to_vec(path: &str) -> io::Result<Vec<u8>> {
    fs::read(path)
}

fn main() {
    match read_binary_file_to_vec("../../0_500.bin") {
        Ok(bytes) => {
            println!("Successfully read {} bytes from the binary file.", bytes.len());
            // Process the bytes here
            for byte in bytes.iter().take(100) { // Print first 10 bytes as an example
                print!("{:02x} ", byte);
            }
            println!();
        }
        Err(e) => {
            eprintln!("Error reading binary file: {}", e);
        }
    }
}
