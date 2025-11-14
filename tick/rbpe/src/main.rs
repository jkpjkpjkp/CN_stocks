use std::fs;
use std::io;
use rayon::prelude::*;
use std::collections::HashMap;

pub fn count_byte_pairs(bytes: &[u8]) -> HashMap<(u8, u8), usize> {
    
    bytes.par_windows(2)  // Parallel iterator over byte pairs
        .map(|window| (window[0], window[1]))  // Map to pairs
        .fold(
            || HashMap::with_capacity(4096),  // Initialize thread-local map
            |mut map, pair| {  // Fold: count pairs locally
                if pair.0 != 0x00 && pair.1 != 0x00 {
                    map.entry(pair).and_modify(|c| *c += 1).or_insert(1);
                }
                map
            }
        )
        .reduce(
            || HashMap::with_capacity(4096),  // Initialize accumulator
            |mut map1, map2| {  // Reduce: merge maps
                for (pair, count) in map2 {
                    map1.entry(pair)
                        .and_modify(|c| *c += count)
                        .or_insert(count);
                }
                map1
            }
        )
}

pub fn most_frequent_pair(bytes: &[u8]) -> Option<((u8, u8), usize)> {
    count_byte_pairs(bytes).into_iter().max_by_key(|(_, count)| *count)
}

fn read_binary_file_to_vec(path: &str) -> io::Result<Vec<u8>> {
    fs::read(path)
}

fn main() {
    let bytes: Vec<u8>;
    let Ok(bytes) = read_binary_file_to_vec("../../0_500.bin") else {
        eprintln!("Error reading binary file");
        return;
    };
    println!("Successfully read {} bytes from the binary file.", bytes.len());
    
    for byte in bytes.iter().take(10) { // Print first 10 bytes as an example
        print!("{:02x} ", byte);
    }
    println!();
    
    let mut data1 = Vec<u16>;
    let mut data2 = Vec<u16>;
    let mut merges = Vec<((u16, u16), u16)>;
    let mut tot = 256;
    for i in 0..1000 {
        let pair = most_frequent_pair(bytes);
        merges.push((pair.unwrap(), tot));
        tot += 1;
        
    }
}
