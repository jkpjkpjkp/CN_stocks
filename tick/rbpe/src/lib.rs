use std::fs;
use std::io;
use std::env;
use std::collections::HashMap;
use std::cmp::Reverse;
use std::bstr::ByteString;

use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Pair(u16, u16);

impl Pair {
    fn new(a: u16, b: u16) -> Self {
        Self(a, b)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MergeCandidate {
    count: usize,
    pair: Pair,
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.count.cmp(&other.count)
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub fn count_byte_pairs_parallel(bytes: &[u16]) -> HashMap<Pair, usize> {
    bytes.par_windows(2)
        .filter(|window| window[0] != 0x00 && window[1] != 0x00) // Skip separators
        .map(|window| Pair::new(window[0], window[1]))
        .fold(
            || HashMap::with_capacity(4096),
            |mut map, pair| {
                *map.entry(pair).or_insert(0) += 1;
                map
            }
        )
        .reduce(
            || HashMap::with_capacity(4096),
            |mut map1, map2| {
                for (pair, count) in map2 {
                    *map1.entry(pair).or_insert(0) += count;
                }
                map1
            }
        )
}

fn merge_pair(ids: &[u16], pair: Pair, new_id: u16) -> Vec<u16> {
    let (a, b) = (pair.0, pair.1);
    let n = ids.len();
    let mut out = Vec::with_capacity(n);
    
    let mut i = 0;
    while i < n {
        if i + 1 < n && ids[i] == a && ids[i + 1] == b {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    
    out
}

fn read_binary_file_to_vec(path: &str) -> io::Result<Vec<u8>> {
    fs::read(path)
}

fn convert_bytes_to_tokens(bytes: &[u8]) -> Vec<u16> {
    bytes.par_iter()
        .map(|&b| b as u16)
        .collect()
}

fn bpe(bytes: &Vec<u8>, next_id: u16 = 256, num_merges: usize = 10) {
    let mut tokens = convert_bytes_to_tokens(&bytes);
    
    // Initialize pair counting
    let mut pair_counts = count_byte_pairs_parallel(&tokens);
    
    let mut merges = Vec::new();
    let mut next_id = next_id;
    
    println!("Starting BPE training with {} tokens...", tokens.len());
    
    let mut token_to_str: HashMap<u16, ByteString> = HashMap::new();
    for i in -63..=63{
        token_to_str[i + 64 as u16] = ByteString::from(vec![i as u8]);
    }

    for i in 0..num_merges {
        if pair_counts.is_empty() {
            println!("No more pairs to merge at iteration {}", i);
            break;
        }
        
        // Get the most frequent pair
        let candidate = pair_counts.iter()
            .max_by_key(|&(_, &count)| count)
            .unwrap()
            .0.clone();
        let current_pair = candidate;
        let current_count = pair_counts[&current_pair];
        
        // Verify the count is still accurate (due to delta updates)
        if pair_counts.get(&current_pair).map(|&c| c) != Some(current_count) {
            continue; // Skip stale entries
        }
        
        if current_count <= 1 {
            println!("Stopping at iteration {} - no pairs with frequency > 1", i);
            break;
        }
        
        // Perform the merge
        tokens = merge_pair(&tokens, current_pair, next_id);
        pair_counts = count_byte_pairs_parallel(&tokens);
        
        // Record the merge
        merges.push((current_pair, next_id));
        token_to_str[next_id] = &token_to_str[current_pair.0].clone() + &token_to_str[current_pair.1].clone();
        
        println!("Iteration {}: merged pair ({}, {}) -> {} (count: {})", 
                    i, current_pair.0, current_pair.1, next_id, current_count);
        
        next_id += 1;
    }
    
    println!("Completed {} merges. Final vocabulary size: {}", merges.len(), next_id);
    
    // Save merges to file (optional)
    println!("Most frequent merges:");
    for (i, &(pair, id)) in merges.iter().enumerate().take(10) {
        println!("{}: ({}, {}) -> {}", i, pair.0, pair.1, id);
    }
}

fn mian() {
    let bytes: Vec<u8>;
    let Ok(bytes) = read_binary_file_to_vec("../../0_500.bin") else {
        eprintln!("Error reading binary file");
        return;
    };
    println!("Successfully read {} bytes from the binary file.", bytes.len());
    bpe(&bytes)
}