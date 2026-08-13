//! Print an ONNX model's I/O signature and metadata.
//!
//! The SenseVoice exports state their own preprocessing contract in ONNX
//! metadata -- LFR window, CMVN vectors, blank id, language ids -- so this is
//! how you read the numbers the engine has to agree with, rather than copying
//! them out of a blog post. Run it again whenever the model version moves.
//!
//! ORT_DYLIB_PATH=target/debug/lib/libonnxruntime.so \
//!   cargo run -p nevoflux-asr --example dump_model \
//!     --features sensevoice,ort-load-dynamic -- <model.onnx>

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: dump_model <model.onnx>")?;
    let session = ort::session::Session::builder()?.commit_from_file(&path)?;

    println!("== inputs ==");
    for i in session.inputs() {
        println!("  {:<20} {:?}", i.name(), i.dtype());
    }
    println!("== outputs ==");
    for o in session.outputs() {
        println!("  {:<20} {:?}", o.name(), o.dtype());
    }

    println!("== metadata ==");
    let meta = session.metadata()?;
    for key in meta.custom_keys()? {
        let value = meta.custom(&key).unwrap_or_default();
        // The CMVN vectors are 560 floats each; print only their shape and
        // ends, or the useful keys scroll off the screen.
        if value.len() > 120 {
            let parts: Vec<&str> = value.split(',').collect();
            println!(
                "  {key} = [{} values] {} … {}",
                parts.len(),
                parts.first().unwrap_or(&""),
                parts.last().unwrap_or(&"")
            );
        } else {
            println!("  {key} = {value}");
        }
    }
    Ok(())
}
