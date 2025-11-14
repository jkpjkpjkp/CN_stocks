use std::fs;
use std::io;
use rayon::prelude::*;
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;

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

fn build_pair_heap(pair_counts: &HashMap<Pair, usize>) -> BinaryHeap<MergeCandidate> {
    let mut heap = BinaryHeap::with_capacity(pair_counts.len());
    
    for (&pair, &count) in pair_counts {
        if count > 0 {
            heap.push(MergeCandidate { count, pair });
        }
    }
    
    heap
}

fn merge_pair(ids: &[u16], pair: Pair, new_id: u16) -> (Vec<u16>, Vec<(Pair, i32)>) {
    let (a, b) = (pair.0, pair.1);
    let n = ids.len();
    let mut out = Vec::with_capacity(n);
    let mut deltas = Vec::new();
    
    let mut i = 0;
    while i < n {
        if i + 1 < n && ids[i] == a && ids[i + 1] == b {
            // Handle left neighbor
            if i > 0 && ids[i - 1] != 0x00 {
                let left = ids[i - 1];
                // Remove old pairs
                deltas.push((Pair::new(left, a), -1));
                deltas.push((Pair::new(left, b), -1));
                // Add new pair
                deltas.push((Pair::new(left, new_id), 1));
            }
            
            // Remove the merged pair itself
            deltas.push((pair, -1));
            
            // Handle right neighbor
            if i + 2 < n && ids[i + 2] != 0x00 {
                let right = ids[i + 2];
                // Remove old pairs
                deltas.push((Pair::new(b, right), -1));
                // Add new pair
                deltas.push((Pair::new(new_id, right), 1));
            }
            
            out.push(new_id);
            i += 2; // Skip the merged pair
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    
    (out, deltas)
}

fn apply_deltas(pair_counts: &mut HashMap<Pair, usize>, deltas: &[(Pair, i32)]) {
    for &(pair, delta) in deltas {
        let count = pair_counts.entry(pair).or_insert(0);
        if delta < 0 {
            *count = count.saturating_sub((-delta) as usize);
        } else {
            *count = count.saturating_add(delta as usize);
        }
        
        // Clean up zero counts to save memory
        if *count == 0 {
            pair_counts.remove(&pair);
        }
    }
}

fn read_binary_file_to_vec(path: &str) -> io::Result<Vec<u8>> {
    fs::read(path)
}

fn convert_bytes_to_tokens(bytes: &[u8]) -> Vec<u16> {
    bytes.iter()
        .map(|&b| b as u16)
        .collect()
}

fn main() {
    let bytes: Vec<u8>;
    let Ok(bytes) = read_binary_file_to_vec("../../0_500.bin") else {
        eprintln!("Error reading binary file");
        return;
    };
    println!("Successfully read {} bytes from the binary file.", bytes.len());
    
    // Convert bytes to initial token IDs (0-255 for bytes, 256+ for merges)
    let mut tokens = convert_bytes_to_tokens(&bytes);
    
    // Initialize pair counting
    let mut pair_counts = count_byte_pairs_parallel(&tokens);
    
    // Build priority queue for efficient frequent pair lookup
    let mut heap = build_pair_heap(&pair_counts);
    
    let mut merges = Vec::new();
    let mut next_id = 256;
    let target_merges = 1000;
    
    println!("Starting BPE training with {} tokens...", tokens.len());
    
    for i in 0..target_merges {
        if heap.is_empty() {
            println!("No more pairs to merge at iteration {}", i);
            break;
        }
        
        // Get the most frequent pair
        let candidate = heap.pop().unwrap();
        let current_pair = candidate.pair;
        let current_count = candidate.count;
        
        // Verify the count is still accurate (due to delta updates)
        if pair_counts.get(&current_pair).map(|&c| c) != Some(current_count) {
            continue; // Skip stale entries
        }
        
        if current_count <= 1 {
            println!("Stopping at iteration {} - no pairs with frequency > 1", i);
            break;
        }
        
        // Perform the merge
        let (new_tokens, deltas) = merge_pair(&tokens, current_pair, next_id);
        tokens = new_tokens;
        
        // Apply delta updates to pair counts
        apply_deltas(&mut pair_counts, &deltas);
        
        // Update the heap with affected pairs
        for &(pair, delta) in &deltas {
            if delta > 0 {
                if let Some(&count) = pair_counts.get(&pair) {
                    heap.push(MergeCandidate { count, pair });
                }
            }
        }
        
        // Record the merge
        merges.push((current_pair, next_id));
        
        if i % 100 == 0 {
            println!("Iteration {}: merged pair ({}, {}) -> {} (count: {})", 
                     i, current_pair.0, current_pair.1, next_id, current_count);
        }
        
        next_id += 1;
    }
    
    println!("Completed {} merges. Final vocabulary size: {}", merges.len(), next_id);
    
    // Save merges to file (optional)
    println!("Most frequent merges:");
    for (i, &(pair, id)) in merges.iter().enumerate().take(10) {
        println!("{}: ({}, {}) -> {}", i, pair.0, pair.1, id);
    }
}