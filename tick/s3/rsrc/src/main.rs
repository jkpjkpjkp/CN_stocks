use parquet::file::reader::FileReader;


fn main() {
    // read parquet with `parquet`
    let file = std::fs::File::open("/dev/shm/tmp.parquet").unwrap();
    let reader = FileReader::new(file).unwrap();

    // 
}
