use std::error::Error;
use std::env;

fn main() -> Result<(), Box<dyn Error>> {

    let mut args = env::args();
    if args.len() != 2 { return Ok(()); }
    let img = args.nth(1).unwrap();
    println!("image={}", &img);    

    nftimg::convert(&img)?;
    
    Ok(())
}


