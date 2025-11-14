use rayon::prelude::*;
use std::collections::HashMap;

pub fn count_byte_pairs(bytes: &[u8]) -> HashMap<(u8, u8), usize> {
    
    bytes.par_windows(2)  // Parallel iterator over byte pairs
        .map(|window| (window[0], window[1]))  // Map to pairs
        .fold(
            || HashMap::with_capacity(4096),  // Initialize thread-local map
            |mut map, pair| {  // Fold: count pairs locally
                map.entry(pair).and_modify(|c| *c += 1).or_insert(1);
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_most_frequent_pair() {
        let bytes = b"hellololo world";
        assert_eq!(most_frequent_pair(bytes), Some(((b'l', b'o'), 3)));
    }

    #[test]
    fn test_count_byte_pairs() {
        let bytes = b"hellololo world";
        let expected = HashMap::from([
            ((b'h', b'e'), 1),
            ((b'e', b'l'), 1),
            ((b'l', b'l'), 1),
            ((b'l', b'o'), 3),
            ((b'o', b'l'), 2),
            ((b'o', b' '), 1),
            ((b' ', b'w'), 1),
            ((b'w', b'o'), 1),
            ((b'o', b'r'), 1),
            ((b'r', b'l'), 1),
            ((b'l', b'd'), 1),
        ]);
        assert_eq!(count_byte_pairs(bytes), expected);
    }
}